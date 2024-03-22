# BSD 2-Clause License #
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

import copy
import functools
import os
import os.path as osp
import pathlib
import shutil
import sys
import time
import typing as t
from os import makedirs

from smartsim._core.utils.helpers import create_lockfile_name
from smartsim.error import SSInternalError
from smartsim.settings.containers import Container, Singularity

from .._core.config import CONFIG
from .._core.utils.helpers import (
    expand_exe_path,
    fmt_dict,
    get_base_36_repr,
    is_valid_cmd,
)
from ..entity.dbobject import DBModel, DBScript
from ..error.errors import UnproxyableStepError
from ..log import get_logger

_RunSettingsT = t.TypeVar("_RunSettingsT", bound="RunSettings")
logger = get_logger(__name__)


def proxyable_launch_cmd_jp(
    fn: t.Callable[[_RunSettingsT], t.List[str]],
) -> t.Callable[[_RunSettingsT], t.List[str]]:
    @functools.wraps(fn)
    def _get_launch_cmd_jp(self: _RunSettingsT, name, path) -> t.List[str]:

        name = self._create_unique_name(name)

        # local launch for now:
        managed = False
        original_cmd_list = fn(self, name, path)

        if not CONFIG.telemetry_enabled:
            return original_cmd_list

        # managed --> wlm that takes care
        # unmanaged --> we are executed them as prcoesses (local launcher)
        # ... need to change where managed is being passed ...
        if managed:
            raise UnproxyableStepError(
                f"Attempting to proxy managed step of type {type(self)}"
                "through the unmanaged step proxy entry point"
            )

        proxy_module = "smartsim._core.entrypoints.indirect"
        etype = self.meta["entity_type"]
        status_dir = self.meta["status_dir"]
        #  encoded_cmd = encode_cmd(original_cmd_list)
        encoded_cmd = original_cmd_list  # not encoding for now

        # NOTE: this is NOT safe. should either 1) sign cmd and verify OR 2)
        #       serialize step and let the indirect entrypoint rebuild the
        #       cmd... for now, test away...
        return [
            sys.executable,
            "-m",
            proxy_module,
            "+name",
            name,
            "+command",
            encoded_cmd,
            "+entity_type",
            etype,
            "+telemetry_dir",
            status_dir,
            "+working_dir",
            path,
        ]

    return _get_launch_cmd_jp


# fmt: off
class SettingsBase:
    ...
# fmt: on


# pylint: disable=too-many-public-methods
class RunSettings(SettingsBase):
    # pylint: disable=unused-argument

    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_command: str = "",
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        container: t.Optional[Container] = None,
        meta: t.Dict[str, str] = None,
        launch_cmd: t.Optional[str] = None,
        **_kwargs: t.Any,
    ) -> None:
        """Run parameters for a ``Model``

        The base ``RunSettings`` class should only be used with the `local`
        launcher on single node, workstations, or laptops.

        If no ``run_command`` is specified, the executable will be launched
        locally.

        ``run_args`` passed as a dict will be interpreted literally for
        local ``RunSettings`` and added directly to the ``run_command``
        e.g. run_args = {"-np": 2} will be "-np 2"

        Example initialization

        .. highlight:: python
        .. code-block:: python

            rs = RunSettings("echo", "hello", "mpirun", run_args={"-np": "2"})

        :param exe: executable to run
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_command: launch binary (e.g. "srun"), defaults to empty str
        :type run_command: str, optional
        :param run_args: arguments for run command (e.g. `-np` for `mpiexec`),
            defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        :param container: container type for workload (e.g. "singularity"),
            defaults to None
        :type container: Container, optional
        """
        # Do not expand executable if running within a container
        self.exe = [exe] if container else [expand_exe_path(exe)]
        self.exe_args = exe_args or []
        self.run_args = run_args or {}
        self.env_vars = env_vars or {}
        self.container = container
        self._run_command = run_command
        self.in_batch = False
        self.colocated_db_settings: t.Optional[
            t.Dict[
                str,
                t.Union[
                    bool,
                    int,
                    str,
                    None,
                    t.List[str],
                    t.Iterable[t.Union[int, t.Iterable[int]]],
                    t.List[DBModel],
                    t.List[DBScript],
                    t.Dict[str, t.Union[int, None]],
                    t.Dict[str, str],
                ],
            ]
        ] = None

        self.meta = meta or {}
        self.launch_cmd = launch_cmd

    @property
    def get_launch_cmd(self):
        return self.launch_cmd

    def _create_job_step_rs(self, entity, telemetry_dir: t.Optional[t.Any] = None):
        """Create job steps on the run settings

        :param entity: an entity to create a step for
        :type entity: SmartSimEntity
        :param telemetry_dir: Path to a directory in which the job step
                               may write telemetry events
        :type telemetry_dir: pathlib.Path
        :return: the job step
        :rtype: Step
        """
        # if isinstance(entity, Model):
        #    self._prep_entity_client_env(entity)

        # step = self._launcher.create_step(entity.name, entity.path, entity.run_settings)

        self.meta["entity_type"] = str(type(entity).__name__).lower()
        self.meta["status_dir"] = str(telemetry_dir / entity.name)
        step = self.get_launch_cmd_jp(entity.name, entity.path)

        return step

    @proxyable_launch_cmd_jp
    def get_launch_cmd_jp(self, name, path) -> t.List[str]:
        cmd = []

        # Add run command and args if user specified
        # default is no run command for local job steps
        if self.run_command:
            cmd.append(self.run_command)
            run_args = self.format_run_args()
            cmd.extend(run_args)

        if self.colocated_db_settings:
            # Replace the command with the entrypoint wrapper script
            if not (bash := shutil.which("bash")):
                raise RuntimeError("Unable to locate bash interpreter")

            launch_script_path = self.get_colocated_launch_script(name, path)
            cmd.extend([bash, launch_script_path])

        container = self.container
        if container and isinstance(container, Singularity):
            # pylint: disable-next=protected-access
            cmd += container._container_cmds(self.cwd)

        # build executable
        cmd.extend(self.exe)

        if self.exe_args:
            cmd.extend(self.exe_args)

        self.launch_cmd = cmd
        return cmd

    def get_colocated_launch_script(self, entity_name, path) -> str:
        # prep step for colocated launch if specifed in run settings
        script_path = self.get_step_file(
            entity_name,
            path,
            script_name=osp.join(".smartsim", f"colocated_launcher_{entity_name}.sh"),
        )
        makedirs(osp.dirname(script_path), exist_ok=True)

        db_settings = {}
        # jpnote look at this
        # if isinstance(self.step_settings, RunSettings):
        #     db_settings = self.step_settings.colocated_db_settings or {}

        if isinstance(self, RunSettings):
            db_settings = self.colocated_db_settings or {}

        # db log file causes write contention and kills performance so by
        # default we turn off logging unless user specified debug=True
        if db_settings.get("debug", False):
            db_log_file = self.get_step_file(entity_name, path, ending="-db.log")
        else:
            db_log_file = "/dev/null"

        # write the colocated wrapper shell script to the directory for this
        # entity currently being prepped to launch
        self.write_colocated_launch_script(script_path, db_log_file, db_settings)
        return script_path

    def _build_db_model_cmd(self, db_models: t.List[DBModel]) -> t.List[str]:
        cmd = []
        for db_model in db_models:
            cmd.append("+db_model")
            cmd.append(f"--name={db_model.name}")

            # Here db_model.file is guaranteed to exist
            # because we don't allow the user to pass a serialized DBModel
            cmd.append(f"--file={db_model.file}")

            cmd.append(f"--backend={db_model.backend}")
            cmd.append(f"--device={db_model.device}")
            cmd.append(f"--devices_per_node={db_model.devices_per_node}")
            cmd.append(f"--first_device={db_model.first_device}")
            if db_model.batch_size:
                cmd.append(f"--batch_size={db_model.batch_size}")
            if db_model.min_batch_size:
                cmd.append(f"--min_batch_size={db_model.min_batch_size}")
            if db_model.min_batch_timeout:
                cmd.append(f"--min_batch_timeout={db_model.min_batch_timeout}")
            if db_model.tag:
                cmd.append(f"--tag={db_model.tag}")
            if db_model.inputs:
                cmd.append("--inputs=" + ",".join(db_model.inputs))
            if db_model.outputs:
                cmd.append("--outputs=" + ",".join(db_model.outputs))

        return cmd

    def _build_db_script_cmd(self, db_scripts: t.List[DBScript]) -> t.List[str]:
        cmd = []
        for db_script in db_scripts:
            cmd.append("+db_script")
            cmd.append(f"--name={db_script.name}")
            if db_script.func:
                # Notice that here db_script.func is guaranteed to be a str
                # because we don't allow the user to pass a serialized function
                sanitized_func = db_script.func.replace("\n", "\\n")
                if not (
                    sanitized_func.startswith("'")
                    and sanitized_func.endswith("'")
                    or (sanitized_func.startswith('"') and sanitized_func.endswith('"'))
                ):
                    sanitized_func = '"' + sanitized_func + '"'
                cmd.append(f"--func={sanitized_func}")
            elif db_script.file:
                cmd.append(f"--file={db_script.file}")
            cmd.append(f"--device={db_script.device}")
            cmd.append(f"--devices_per_node={db_script.devices_per_node}")
            cmd.append(f"--first_device={db_script.first_device}")
        return cmd

    def _build_colocated_wrapper_cmd(
        self,
        db_log: str,
        cpus: int = 1,
        rai_args: t.Optional[t.Dict[str, str]] = None,
        extra_db_args: t.Optional[t.Dict[str, str]] = None,
        port: int = 6780,
        ifname: t.Optional[t.Union[str, t.List[str]]] = None,
        custom_pinning: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> str:
        """Build the command use to run a colocated DB application

        :param db_log: log file for the db
        :type db_log: str
        :param cpus: db cpus, defaults to 1
        :type cpus: int, optional
        :param rai_args: redisai args, defaults to None
        :type rai_args: dict[str, str], optional
        :param extra_db_args: extra redis args, defaults to None
        :type extra_db_args: dict[str, str], optional
        :param port: port to bind DB to
        :type port: int
        :param ifname: network interface(s) to bind DB to
        :type ifname: str | list[str], optional
        :param db_cpu_list: The list of CPUs that the database should be limited to
        :type db_cpu_list: str, optional
        :return: the command to run
        :rtype: str
        """
        # pylint: disable=too-many-locals

        # create unique lockfile name to avoid symlink vulnerability
        # this is the lockfile all the processes in the distributed
        # application will try to acquire. since we use a local tmp
        # directory on the compute node, only one process can acquire
        # the lock on the file.
        lockfile = create_lockfile_name()

        # create the command that will be used to launch the
        # database with the python entrypoint for starting
        # up the backgrounded db process

        cmd = [
            sys.executable,
            "-m",
            "smartsim._core.entrypoints.colocated",
            "+lockfile",
            lockfile,
            "+db_cpus",
            str(cpus),
        ]
        # Add in the interface if using TCP/IP
        if ifname:
            if isinstance(ifname, str):
                ifname = [ifname]
            cmd.extend(["+ifname", ",".join(ifname)])
        cmd.append("+command")
        # collect DB binaries and libraries from the config

        db_cmd = []
        if custom_pinning:
            db_cmd.extend(["taskset", "-c", custom_pinning])
        db_cmd.extend(
            [CONFIG.database_exe, CONFIG.database_conf, "--loadmodule", CONFIG.redisai]
        )

        # add extra redisAI configurations
        for arg, value in (rai_args or {}).items():
            if value:
                # RAI wants arguments for inference in all caps
                # ex. THREADS_PER_QUEUE=1
                db_cmd.append(f"{arg.upper()} {str(value)}")

        db_cmd.extend(["--port", str(port)])

        # Add socket and permissions for UDS
        unix_socket = kwargs.get("unix_socket", None)
        socket_permissions = kwargs.get("socket_permissions", None)

        if unix_socket and socket_permissions:
            db_cmd.extend(
                [
                    "--unixsocket",
                    str(unix_socket),
                    "--unixsocketperm",
                    str(socket_permissions),
                ]
            )
        elif bool(unix_socket) ^ bool(socket_permissions):
            raise SSInternalError(
                "`unix_socket` and `socket_permissions` must both be defined or undefined."
            )

        db_cmd.extend(
            ["--logfile", db_log]
        )  # usually /dev/null, unless debug was specified
        if extra_db_args:
            for db_arg, value in extra_db_args.items():
                # replace "_" with "-" in the db_arg because we use kwargs
                # for the extra configurations and Python doesn't allow a hyphen
                # in a variable name. All redis and KeyDB configuration options
                # use hyphens in their names.
                db_arg = db_arg.replace("_", "-")
                db_cmd.extend([f"--{db_arg}", value])

        db_models = kwargs.get("db_models", None)
        if db_models:
            db_model_cmd = self._build_db_model_cmd(db_models)
            db_cmd.extend(db_model_cmd)

        db_scripts = kwargs.get("db_scripts", None)
        if db_scripts:
            db_script_cmd = self._build_db_script_cmd(db_scripts)
            db_cmd.extend(db_script_cmd)

        # run colocated db in the background
        db_cmd.append("&")

        cmd.extend(db_cmd)
        return " ".join(cmd)

    def write_colocated_launch_script(
        self, file_name: str, db_log: str, colocated_settings: t.Dict[str, t.Any]
    ) -> None:
        """Write the colocated launch script

        This file will be written into the cwd of the step that
        is created for this entity.

        :param file_name: name of the script to write
        :type file_name: str
        :param db_log: log file for the db
        :type db_log: str
        :param colocated_settings: db settings from entity run_settings
        :type colocated_settings: dict[str, Any]
        """

        colocated_cmd = self._build_colocated_wrapper_cmd(db_log, **colocated_settings)

        with open(file_name, "w", encoding="utf-8") as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write("set -e\n\n")

            script_file.write("Cleanup () {\n")
            script_file.write("if ps -p $DBPID > /dev/null; then\n")
            script_file.write("\tkill -15 $DBPID\n")
            script_file.write("fi\n}\n\n")

            # run cleanup after all exitcodes
            script_file.write("trap Cleanup exit\n\n")

            # force entrypoint to write some debug information to the
            # STDOUT of the job
            if colocated_settings["debug"]:
                script_file.write("export SMARTSIM_LOG_LEVEL=debug\n")

            script_file.write(f"{colocated_cmd}\n")
            script_file.write("DBPID=$!\n\n")

            # Write the actual launch command for the app
            script_file.write("$@\n\n")

    ## get the output and err files ??

    def get_output_files(self, name, path) -> t.Tuple[str, str]:
        """Return two paths to error and output files based on cwd"""
        output = self.get_step_file(name, path, ending=".out")
        error = self.get_step_file(name, path, ending=".err")
        return output, error

    def get_step_file(
        self,
        name,
        path,
        ending: str = ".sh",
        script_name: t.Optional[str] = None,
    ) -> str:
        """Get the name for a file/script created by the step class

        Used for Batch scripts, mpmd scripts, etc.
        """
        if script_name:
            script_name = script_name if "." in script_name else script_name + ending
            return osp.join(path, script_name)
        return osp.join(path, name + ending)

    def _set_env(self) -> t.Dict[str, str]:
        env = os.environ.copy()
        if self.env_vars:
            for k, v in self.env_vars.items():
                env[k] = v or ""
        return env

    @property
    def exe_args(self) -> t.Union[str, t.List[str]]:
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Union[str, t.List[str], None]) -> None:
        self._exe_args = self._build_exe_args(value)

    @property
    def run_args(self) -> t.Dict[str, t.Union[int, str, float, None]]:
        return self._run_args

    @run_args.setter
    def run_args(self, value: t.Dict[str, t.Union[int, str, float, None]]) -> None:
        self._run_args = copy.deepcopy(value)

    @property
    def env_vars(self) -> t.Dict[str, t.Optional[str]]:
        return self._env_vars

    @env_vars.setter
    def env_vars(self, value: t.Dict[str, t.Optional[str]]) -> None:
        self._env_vars = copy.deepcopy(value)

    # To be overwritten by subclasses. Set of reserved args a user cannot change
    reserved_run_args = set()  # type: set[str]

    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        :param nodes: number of nodes to run with
        :type nodes: int
        """
        logger.warning(
            (
                "Node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks to launch

        :param tasks: number of tasks to launch
        :type tasks: int
        """
        logger.warning(
            (
                "Task specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        logger.warning(
            (
                "Task per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_task_map(self, task_mapping: str) -> None:
        """Set a task mapping

        :param task_mapping: task mapping
        :type task_mapping: str
        """
        logger.warning(
            (
                "Task mapping specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus per task

        :param cpus_per_task: number of cpus per task
        :type cpus_per_task: int
        """
        logger.warning(
            (
                "CPU per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        """
        logger.warning(
            (
                "Hostlist specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to specify the hostlist for this job

        :param file_path: Path to the hostlist file
        :type file_path: str
        """
        logger.warning(
            (
                "Hostlist from file specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :type host_list: str | list[str]
        """
        logger.warning(
            (
                "Excluded host list specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """Set the cores to which MPI processes are bound

        :param bindings: List specifing the cores to which MPI processes are bound
        :type bindings: list[int] | int
        """
        logger.warning(
            (
                "CPU binding specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Set the amount of memory required per node in megabytes

        :param memory_per_node: Number of megabytes per node
        :type memory_per_node: int
        """
        logger.warning(
            (
                "Memory per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        :param verbose: Whether the job should be run verbosely
        :type verbose: bool
        """
        logger.warning(
            (
                "Verbose specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        :param quiet: Whether the job should be run quietly
        :type quiet: bool
        """
        logger.warning(
            (
                "Quiet specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_broadcast(self, dest_path: t.Optional[str] = None) -> None:
        """Copy executable file to allocated compute nodes

        :param dest_path: Path to copy an executable file
        :type dest_path: str | None
        """
        logger.warning(
            (
                "Broadcast specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_time(self, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        """Automatically format and set wall time

        :param hours: number of hours to run job
        :type hours: int
        :param minutes: number of minutes to run job
        :type minutes: int
        :param seconds: number of seconds to run job
        :type seconds: int
        """
        return self.set_walltime(
            self._fmt_walltime(int(hours), int(minutes), int(seconds))
        )

    @staticmethod
    def _fmt_walltime(hours: int, minutes: int, seconds: int) -> str:
        """Convert hours, minutes, and seconds into valid walltime format

        By defualt the formatted wall time is the total number of seconds.

        :param hours: number of hours to run job
        :type hours: int
        :param minutes: number of minutes to run job
        :type minutes: int
        :param seconds: number of seconds to run job
        :type seconds: int
        :returns: Formatted walltime
        :rtype: str
        """
        time_ = hours * 3600
        time_ += minutes * 60
        time_ += seconds
        return str(time_)

    def set_walltime(self, walltime: str) -> None:
        """Set the formatted walltime

        :param walltime: Time in format required by launcher``
        :type walltime: str
        """
        logger.warning(
            (
                "Walltime specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_binding(self, binding: str) -> None:
        """Set binding

        :param binding: Binding
        :type binding: str
        """
        logger.warning(
            (
                "binding specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_mpmd_preamble(self, preamble_lines: t.List[str]) -> None:
        """Set preamble to a file to make a job MPMD

        :param preamble_lines: lines to put at the beginning of a file.
        :type preamble_lines: list[str]
        """
        logger.warning(
            (
                "MPMD preamble specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def make_mpmd(self, settings: RunSettings) -> None:
        """Make job an MPMD job

        :param settings: ``RunSettings`` instance
        :type settings: RunSettings
        """
        logger.warning(
            (
                "Make MPMD specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    @property
    def run_command(self) -> t.Optional[str]:
        """Return the launch binary used to launch the executable

        Attempt to expand the path to the executable if possible

        :returns: launch binary e.g. mpiexec
        :type: str | None
        """
        cmd = self._run_command

        if cmd:
            if is_valid_cmd(cmd):
                # command is valid and will be expanded
                return expand_exe_path(cmd)
            # command is not valid, so return it as is
            # it may be on the compute nodes but not local machine
            return cmd
        # run without run command
        return None

    def update_env(self, env_vars: t.Dict[str, t.Union[str, int, float, bool]]) -> None:
        """Update the job environment variables

        To fully inherit the current user environment, add the
        workload-manager-specific flag to the launch command through the
        :meth:`add_exe_args` method. For example, ``--export=ALL`` for
        slurm, or ``-V`` for PBS/aprun.


        :param env_vars: environment variables to update or add
        :type env_vars: dict[str, Union[str, int, float, bool]]
        :raises TypeError: if env_vars values cannot be coerced to strings
        """
        val_types = (str, int, float, bool)
        # Coerce env_vars values to str as a convenience to user
        for env, val in env_vars.items():
            if not isinstance(val, val_types):
                raise TypeError(
                    f"env_vars[{env}] was of type {type(val)}, not {val_types}"
                )

            self.env_vars[env] = str(val)

    def add_exe_args(self, args: t.Union[str, t.List[str]]) -> None:
        """Add executable arguments to executable

        :param args: executable arguments
        :type args: str | list[str]
        """
        args = self._build_exe_args(args)
        self._exe_args.extend(args)

    def set(
        self, arg: str, value: t.Optional[str] = None, condition: bool = True
    ) -> None:
        """Allows users to set individual run arguments.

        A method that allows users to set run arguments after object
        instantiation. Does basic formatting such as stripping leading dashes.
        If the argument has been set previously, this method will log warning
        but ultimately comply.

        Conditional expressions may be passed to the conditional parameter. If the
        expression evaluates to True, the argument will be set. In not an info
        message is logged and no further operation is performed.

        Basic Usage

        .. highlight:: python
        .. code-block:: python

            rs = RunSettings("python")
            rs.set("an-arg", "a-val")
            rs.set("a-flag")
            rs.format_run_args()  # returns ["an-arg", "a-val", "a-flag", "None"]

        Slurm Example with Conditional Setting

        .. highlight:: python
        .. code-block:: python

            import socket

            rs = SrunSettings("echo", "hello")
            rs.set_tasks(1)
            rs.set("exclusive")

            # Only set this argument if condition param evals True
            # Otherwise log and NOP
            rs.set("partition", "debug",
                   condition=socket.gethostname()=="testing-system")

            rs.format_run_args()
            # returns ["exclusive", "None", "partition", "debug"] iff
              socket.gethostname()=="testing-system"
            # otherwise returns ["exclusive", "None"]

        :param arg: name of the argument
        :type arg: str
        :param value: value of the argument
        :type value: str | None
        :param conditon: set the argument if condition evaluates to True
        :type condition: bool
        """
        if not isinstance(arg, str):
            raise TypeError("Argument name should be of type str")
        if value is not None and not isinstance(value, str):
            raise TypeError("Argument value should be of type str or None")
        arg = arg.strip().lstrip("-")

        if not condition:
            logger.info(f"Could not set argument '{arg}': condition not met")
            return
        if arg in self.reserved_run_args:
            logger.warning(
                (
                    f"Could not set argument '{arg}': "
                    f"it is a reserved arguement of '{type(self).__name__}'"
                )
            )
            return

        if arg in self.run_args and value != self.run_args[arg]:
            logger.warning(f"Overwritting argument '{arg}' with value '{value}'")
        self.run_args[arg] = value

    @staticmethod
    def _build_exe_args(exe_args: t.Optional[t.Union[str, t.List[str]]]) -> t.List[str]:
        """Check and convert exe_args input to a desired collection format"""
        if not exe_args:
            return []

        if isinstance(exe_args, list):
            exe_args = copy.deepcopy(exe_args)

        if not (
            isinstance(exe_args, str)
            or (
                isinstance(exe_args, list)
                and all(isinstance(arg, str) for arg in exe_args)
            )
        ):
            raise TypeError("Executable arguments were not a list of str or a str.")

        if isinstance(exe_args, str):
            return exe_args.split()

        return exe_args

    def format_run_args(self) -> t.List[str]:
        """Return formatted run arguments

        For ``RunSettings``, the run arguments are passed
        literally with no formatting.

        :return: list run arguments for these settings
        :rtype: list[str]
        """
        formatted = []
        for arg, value in self.run_args.items():
            formatted.append(arg)
            formatted.append(str(value))
        return formatted

    def format_env_vars(self) -> t.List[str]:
        """Build environment variable string

        :returns: formatted list of strings to export variables
        :rtype: list[str]
        """
        formatted = []
        for key, val in self.env_vars.items():
            if val is None:
                formatted.append(f"{key}=")
            else:
                formatted.append(f"{key}={val}")
        return formatted

    def __str__(self) -> str:  # pragma: no-cover
        string = f"Executable: {self.exe[0]}\n"
        string += f"Executable Arguments: {' '.join((self.exe_args))}"
        if self.run_command:
            string += f"\nRun Command: {self.run_command}"
        if self.run_args:
            string += f"\nRun Arguments:\n{fmt_dict(self.run_args)}"
        if self.colocated_db_settings:
            string += "\nCo-located Database: True"
        return string

    @staticmethod
    def _create_unique_name(entity_name: str) -> str:
        step_name = entity_name + "-" + get_base_36_repr(time.time_ns())
        return step_name


class BatchSettings(SettingsBase):
    def __init__(
        self,
        batch_cmd: str,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        meta: t.Dict[str, str] = (None,),
        **kwargs: t.Any,
    ) -> None:
        self._batch_cmd = batch_cmd
        self.batch_args = batch_args or {}
        self._preamble: t.List[str] = []
        self.set_nodes(kwargs.get("nodes", None))
        self.set_walltime(kwargs.get("time", None))
        self.set_queue(kwargs.get("queue", None))
        self.set_account(kwargs.get("account", None))

        self.meta = meta or {}

    # JPNOTE: do I need to add anything to here, or just in the slurm/pbs settings?

    @property
    def batch_cmd(self) -> str:
        """Return the batch command

        Tests to see if we can expand the batch command
        path. If we can, then returns the expanded batch
        command. If we cannot, returns the batch command as is.

        :returns: batch command
        :type: str
        """
        if is_valid_cmd(self._batch_cmd):
            return expand_exe_path(self._batch_cmd)

        return self._batch_cmd

    @property
    def batch_args(self) -> t.Dict[str, t.Optional[str]]:
        return self._batch_args

    @batch_args.setter
    def batch_args(self, value: t.Dict[str, t.Optional[str]]) -> None:
        self._batch_args = copy.deepcopy(value) if value else {}

    def get_batch_launch_cmd(self): ...

    def _create_batch_job_step_rs(
        self, entity_list, telemetry_dir: t.Optional[t.Any] = None
    ):
        NotImplementedError

        ## i think I just need everything for creating the launch command..
        ## becuase instead of that stuff being in the

        # dont actually want to move all of that outside the controller

        # ## will not go into here at all
        # ## will go directly into slurm settings
        # ## function has the same name but is called by
        # ## batch_settings.name
        # ## where to I save it, can I have a getter fnct

        # print("DOES IT EVER come into here? ")  # doesn't actually fall into here?
        # # goes straight to slurm.. hmmmm
        # if not entity_list.batch_settings:
        #     raise ValueError(
        #         "EntityList must have batch settings to be launched as batch"
        #     )

        # telemetry_dir = telemetry_dir / entity_list.name
        # # batch_step = self._launcher.create_step(
        # #     entity_list.name, entity_list.path, entity_list.batch_settings
        # # )
        # # dont create the step?
        # ## what is this going into???
        # self.meta["entity_type"] = str(type(entity_list).__name__).lower()
        # self.meta["status_dir"] = str(telemetry_dir / entity_list.name)

        # substeps = []
        # for entity in entity_list.entities:
        #     # tells step creation not to look for an allocation
        #     entity.run_settings.in_batch = True
        #     step = self._create_job_step_rs(entity, telemetry_dir)
        #     substeps.append(step)
        #     batch_step.add_to_batch(step)
        # return batch_step, substeps

        # self.meta["entity_type"] = str(type(entity).__name__).lower()
        # self.meta["status_dir"] = str(telemetry_dir / entity.name)

        # print("where is it getting the script?")

    # # create_step(..pass in batch settings )
    # # add to batch stuff

    # # change to batch cmd ...
    # #  @proxyable_launch_cmd_jp
    # def launch_cmd_batch(self) -> t.List[str]:
    #     cmd = []

    #     # Add run command and args if user specified
    #     # default is no run command for local job steps
    #     if self._batch_cmd:
    #         cmd.append(self._batch_cmd)
    #         run_args = self.format_batch_args()
    #         cmd.extend(run_args)

    #     # if self.colocated_db_settings:
    #     #     # Replace the command with the entrypoint wrapper script
    #     #     if not (bash := shutil.which("bash")):
    #     #         raise RuntimeError("Unable to locate bash interpreter")

    #     #     launch_script_path = self.get_colocated_launch_script()
    #     #     cmd.extend([bash, launch_script_path])

    #     # container = self.container
    #     # if container and isinstance(container, Singularity):
    #     #     # pylint: disable-next=protected-access
    #     #     cmd += container._container_cmds(self.cwd)

    #     # build executable
    #     cmd.extend(self.exe)
    #     if self.exe_args:
    #         cmd.extend(self.exe_args)
    #     return cmd

    #####
    # launcher.py
    #     every launcher utilizing this interface must have a map
    #  of supported RunSettings types (see slurmLauncher.py for ex)
    #    def create_step(

    ### do I need to know this stuff to create the launch cmd?  ###
    #         # RunSettings types supported by this launcher

    # @property
    # def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
    #     # RunSettings types supported by this launcher
    #     return {
    #         SrunSettings: SrunStep,
    #         SbatchSettings: SbatchStep,
    #         MpirunSettings: MpirunStep,
    #         MpiexecSettings: MpiexecStep,
    #         OrterunSettings: OrterunStep,
    #         RunSettings: LocalStep,
    #     }

    # def get_step_file(
    #     self,
    #     name,
    #     path,
    #     ending: str = ".sh",
    #     script_name: t.Optional[str] = None,
    # ) -> str:
    #     """Get the name for a file/script created by the step class

    #     Used for Batch scripts, mpmd scripts, etc.
    #     """
    #     print(path)
    #     print(name)
    #     if script_name:
    #         script_name = script_name if "." in script_name else script_name + ending
    #         return osp.join(path, script_name)
    #     return osp.join(path, name + ending)

    #     ## get the output and err files ??

    # def get_output_files(self, name, path) -> t.Tuple[str, str]:
    #     """Return two paths to error and output files based on cwd"""
    #     output = self.get_step_file(name, path, ending=".out")
    #     error = self.get_step_file(name, path, ending=".err")
    #     return output, error

    def set_nodes(self, num_nodes: int) -> None:
        raise NotImplementedError

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        raise NotImplementedError

    def set_queue(self, queue: str) -> None:
        raise NotImplementedError

    def set_walltime(self, walltime: str) -> None:
        raise NotImplementedError

    def set_account(self, account: str) -> None:
        raise NotImplementedError

    def format_batch_args(self) -> t.List[str]:
        raise NotImplementedError

    def set_batch_command(self, command: str) -> None:
        """Set the command used to launch the batch e.g. ``sbatch``

        :param command: batch command
        :type command: str
        """
        self._batch_cmd = command

    def add_preamble(self, lines: t.List[str]) -> None:
        """Add lines to the batch file preamble. The lines are just
        written (unmodified) at the beginning of the batch file
        (after the WLM directives) and can be used to e.g.
        start virtual environments before running the executables.

        :param line: lines to add to preamble.
        :type line: str or list[str]
        """
        if isinstance(lines, str):
            self._preamble += [lines]
        elif isinstance(lines, list):
            self._preamble += lines
        else:
            raise TypeError("Expected str or List[str] for lines argument")

    @property
    def preamble(self) -> t.Iterable[str]:
        """Return an iterable of preamble clauses to be prepended to the batch file"""
        return (clause for clause in self._preamble)

    def __str__(self) -> str:  # pragma: no-cover
        string = f"Batch Command: {self._batch_cmd}"
        if self.batch_args:
            string += f"\nBatch arguments:\n{fmt_dict(self.batch_args)}"
        return string
