# BSD 2-Clause License
#
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
import typing as t
from os import path as osp

from .._core.utils.helpers import expand_exe_path
from ..log import get_logger
from .entity import SmartSimEntity
from .files import EntityFiles

if t.TYPE_CHECKING:
    from smartsim.types import TODO

    RunSettings = TODO
    BatchSettings = TODO


logger = get_logger(__name__)
# TODO: Remove this supression when we strip fileds/functionality
#       (run-settings/batch_settings/params_as_args/etc)!
# pylint: disable-next=too-many-public-methods


class Application(SmartSimEntity):
    def __init__(
        self,
        name: str,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.Sequence[str]]] = None,
        files: t.Optional[EntityFiles] = None,
        file_parameters: t.Mapping[str, str] | None = None,
    ) -> None:
        """Initialize an ``Application``

        :param name: name of the application
        :param exe: executable to run
        :param exe_args: executable arguments
        :param files: files to be copied, symlinked, and/or configured prior to
                      execution
        :param file_parameters: parameters and values to be used when configuring
                                files
        """
        super().__init__(name)
        """The name of the application"""
        self._exe = expand_exe_path(exe)
        """The executable to run"""
        self._exe_args = self._build_exe_args(exe_args) or []
        """The executable arguments"""
        self._files = copy.deepcopy(files) if files else None
        """Files to be copied, symlinked, and/or configured prior to execution"""
        self._file_parameters = (
            copy.deepcopy(file_parameters) if file_parameters else {}
        )
        """Parameters and values to be used when configuring files"""
        self._incoming_entities: t.List[SmartSimEntity] = []
        """Entities for which the prefix will have to be known by other entities"""
        self._key_prefixing_enabled = False
        """Unique prefix to avoid key collisions"""

    @property
    def exe(self) -> str:
        """Return executable to run.

        :returns: application executable to run
        """
        return self._exe

    @exe.setter
    def exe(self, value: str) -> None:
        """Set executable to run.

        :param value: executable to run
        """
        self._exe = copy.deepcopy(value)

    @property
    def exe_args(self) -> t.Sequence[str]:
        """Return a list of attached executable arguments.

        :returns: application executable arguments
        """
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Union[str, t.Sequence[str], None]) -> None:  #
        """Set the executable arguments.

        :param value: executable arguments
        """
        self._exe_args = self._build_exe_args(value)

    @property
    def files(self) -> t.Optional[EntityFiles]:
        """Return files to be copied, symlinked, and/or configured prior to
        execution.

        :returns: files
        """
        return self._files

    @files.setter
    def files(self, value: t.Optional[EntityFiles]) -> None:
        """Set files to be copied, symlinked, and/or configured prior to
        execution.

        :param value: files
        """
        self._files = copy.deepcopy(value)

    @property
    def file_parameters(self) -> t.Mapping[str, str]:
        """Return file parameters.

        :returns: application file parameters
        """
        return self._file_parameters

    @file_parameters.setter
    def file_parameters(self, value: t.Mapping[str, str]) -> None:
        """Set the file parameters.

        :param value: file parameters
        """
        self._file_parameters = copy.deepcopy(value)

    @property
    def incoming_entities(self) -> t.List[SmartSimEntity]:
        """Return incoming entities.

        :returns: incoming entities
        """
        return self._incoming_entities

    @incoming_entities.setter
    def incoming_entities(self, value: t.List[SmartSimEntity]) -> None:
        """Set the incoming entities.

        :param value: incoming entities
        """
        self._incoming_entities = copy.deepcopy(value)

    @property
    def key_prefixing_enabled(self) -> bool:
        """Return whether key prefixing is enabled for the application.

        :param value: key prefixing enabled
        """
        return self._key_prefixing_enabled

    @key_prefixing_enabled.setter
    def key_prefixing_enabled(self, value: bool) -> None:
        """Set whether key prefixing is enabled for the application.

        :param value: key prefixing enabled
        """
        self.key_prefixing_enabled = copy.deepcopy(value)

    def add_exe_args(self, args: t.Union[str, t.List[str], None]) -> None:
        """Add executable arguments to executable

        :param args: executable arguments
        """
        args = self._build_exe_args(args)
        self._exe_args.extend(args)

    def attach_generator_files(
        self,
        to_copy: t.Optional[t.List[str]] = None,
        to_symlink: t.Optional[t.List[str]] = None,
        to_configure: t.Optional[t.List[str]] = None,
    ) -> None:
        """Attach files to an entity for generation

        Attach files needed for the entity that, upon generation,
        will be located in the path of the entity.  Invoking this method
        after files have already been attached will overwrite
        the previous list of entity files.

        During generation, files "to_copy" are copied into
        the path of the entity, and files "to_symlink" are
        symlinked into the path of the entity.

        Files "to_configure" are text based application input files where
        parameters for the application are set. Note that only applications
        support the "to_configure" field. These files must have
        fields tagged that correspond to the values the user
        would like to change. The tag is settable but defaults
        to a semicolon e.g. THERMO = ;10;

        :param to_copy: files to copy
        :param to_symlink: files to symlink
        :param to_configure: input files with tagged parameters
        :raises ValueError: if the generator file already exists
        """
        to_copy = to_copy or []
        to_symlink = to_symlink or []
        to_configure = to_configure or []

        # Check that no file collides with the parameter file written
        # by Generator. We check the basename, even though it is more
        # restrictive than what we need (but it avoids relative path issues)
        for strategy in [to_copy, to_symlink, to_configure]:
            if strategy is not None and any(
                osp.basename(filename) == "smartsim_params.txt" for filename in strategy
            ):
                raise ValueError(
                    "`smartsim_params.txt` is a file automatically "
                    + "generated by SmartSim and cannot be ovewritten."
                )
        self.files = EntityFiles(to_configure, to_copy, to_symlink)

    @property
    def attached_files_table(self) -> str:
        """Return a list of attached files as a plain text table

        :returns: String version of table
        """
        if not self.files:
            return "No file attached to this application."
        return str(self.files)

    def print_attached_files(self) -> None:
        """Print a table of the attached files on std out"""
        print(self.attached_files_table)

    def __str__(self) -> str:  # pragma: no cover
        entity_str = "Name: " + self.name + "\n"
        entity_str += "Type: " + self.type + "\n"
        entity_str += "Executable:\n"
        for ex in self.exe:
            entity_str += f"{ex}\n"
        entity_str += "Executable Arguments:\n"
        for ex_arg in self.exe_args:
            entity_str += f"{str(ex_arg)}\n"
        entity_str += f"Entity Files: {self.files}\n"
        entity_str += f"File Parameters: {self.file_parameters}\n"
        entity_str += "Incoming Entities:\n"
        for entity in self.incoming_entities:
            entity_str += f"{entity}\n"
        entity_str += f"Key Prefixing Enabled: {self.key_prefixing_enabled}\n"

        return entity_str

    @staticmethod
    def _build_exe_args(
        exe_args: t.Optional[t.Union[str, t.Sequence[str], None]]
    ) -> t.List[str]:
        """Check and convert exe_args input to a desired collection format
        
        :param exe_args:
        :raises TypeError: if exe_args is not a list of str or str
        """
        if not exe_args:
            return []

        if not (
            isinstance(exe_args, str)
            or (
                isinstance(exe_args, list)
                and all(isinstance(arg, str) for arg in exe_args)
            )
        ):
            raise TypeError("Executable arguments were not a list of str or a str.")

        if isinstance(exe_args, str):
            return copy.deepcopy(exe_args.split())

        return copy.deepcopy(exe_args)