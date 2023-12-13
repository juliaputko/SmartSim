# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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


import psutil
import pytest

import sys
import numpy as np

from smartsim import Experiment, status
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError

from time import sleep
from smartsim.settings.settings import create_run_settings


def test_orc_parameters():
    threads_per_queue = 2
    inter_op_threads = 2
    intra_op_threads = 2
    db = Orchestrator(
        db_nodes=1,
        threads_per_queue=threads_per_queue,
        inter_op_threads=inter_op_threads,
        intra_op_threads=intra_op_threads,
    )
    assert db.queue_threads == threads_per_queue
    assert db.inter_threads == inter_op_threads
    assert db.intra_threads == intra_op_threads

    module_str = db._rai_module
    assert "THREADS_PER_QUEUE" in module_str
    assert "INTRA_OP_PARALLELISM" in module_str
    assert "INTER_OP_PARALLELISM" in module_str


def test_is_not_active():
    db = Orchestrator(db_nodes=1)
    assert not db.is_active()


def test_inactive_orc_get_address():
    db = Orchestrator()
    with pytest.raises(SmartSimError):
        db.get_address()


def test_orc_active_functions(fileutils, wlmutils):
    exp_name = "test_orc_active_functions"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir()

    db = Orchestrator(port=wlmutils.get_test_port())
    db.set_path(test_dir)

    exp.start(db)

    # check if the orchestrator is active
    assert db.is_active()

    # check if the orchestrator can get the address
    correct_address = db.get_address() == ["127.0.0.1:" + str(wlmutils.get_test_port())]
    if not correct_address:
        exp.stop(db)
        assert False

    exp.stop(db)

    assert not db.is_active()

    # check if orchestrator.get_address() raises an exception
    with pytest.raises(SmartSimError):
        db.get_address()


def test_multiple_interfaces(fileutils, wlmutils):
    exp_name = "test_multiple_interfaces"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir()

    net_if_addrs = psutil.net_if_addrs()
    net_if_addrs = [
        net_if_addr for net_if_addr in net_if_addrs if not net_if_addr.startswith("lo")
    ]

    net_if_addrs = ["lo", net_if_addrs[0]]

    db = Orchestrator(port=wlmutils.get_test_port(), interface=net_if_addrs)
    db.set_path(test_dir)

    exp.start(db)

    # check if the orchestrator is active
    assert db.is_active()

    # check if the orchestrator can get the address
    correct_address = db.get_address() == ["127.0.0.1:" + str(wlmutils.get_test_port())]
    if not correct_address:
        exp.stop(db)
        assert False

    exp.stop(db)


def test_catch_local_db_errors():
    # local database with more than one node not allowed
    with pytest.raises(SSUnsupportedError):
        db = Orchestrator(db_nodes=2)

    # Run command for local orchestrator not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(run_command="srun")

    # Batch mode for local orchestrator is not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(batch=True)


#####  PBS  ######


def test_pbs_set_run_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )
    orc.set_run_arg("pes-per-numa-node", "5")
    assert all(
        ["pes-per-numa-node" not in db.run_settings.run_args for db in orc.entities]
    )


def test_pbs_set_batch_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"
    # orc2.set_batch_arg("N", "another_name")
    # assert "N" not in orc2.batch_settings.batch_args


##### Slurm ######


def test_slurm_set_run_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )


def test_slurm_set_batch_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"


@pytest.mark.parametrize(
    "single_cmd",
    [
        pytest.param(True, id="Single MPMD `srun`"),
        pytest.param(False, id="Multiple `srun`s"),
    ],
)
def test_orc_results_in_correct_number_of_shards(single_cmd):
    num_shards = 5
    orc = Orchestrator(
        port=12345,
        launcher="slurm",
        run_command="srun",
        db_nodes=num_shards,
        batch=False,
        single_cmd=single_cmd,
    )
    if single_cmd:
        assert len(orc.entities) == 1
        (node,) = orc.entities
        assert len(node.run_settings.mpmd) == num_shards - 1
    else:
        assert len(orc.entities) == num_shards
        assert all(node.run_settings.mpmd == [] for node in orc.entities)
    assert (
        orc.num_shards == orc.db_nodes == sum(node.num_shards for node in orc.entities)
    )


###### Cobalt ######


def test_cobalt_set_run_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="cobalt",
        run_command="aprun",
    )
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )
    orc.set_run_arg("pes-per-numa-node", "2")
    assert all(
        ["pes-per-numa-node" not in db.run_settings.run_args for db in orc.entities]
    )


def test_cobalt_set_batch_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="cobalt",
        run_command="aprun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface="lo",
        launcher="cobalt",
        run_command="aprun",
    )
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"
    orc2.set_batch_arg("outputprefix", "new_output/")
    assert "outputprefix" not in orc2.batch_settings.batch_args


###### LSF ######


def test_catch_orc_errors_lsf(wlmutils):
    with pytest.raises(SSUnsupportedError):
        orc = Orchestrator(
            wlmutils.get_test_port(),
            db_nodes=2,
            db_per_host=2,
            batch=False,
            launcher="lsf",
            run_command="jsrun",
        )

    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("P", "MYPROJECT")


def test_lsf_set_run_args(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )
    orc.set_run_arg("l", "gpu-gpu")
    assert all(["l" not in db.run_settings.run_args for db in orc.entities])


def test_lsf_set_batch_args(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )

    assert orc.batch_settings.batch_args["m"] == '"batch host1 host2"'
    orc.set_batch_arg("D", "102400000")
    assert orc.batch_settings.batch_args["D"] == "102400000"


###### write tests for all model cases ######
# ------------------------------------------------------------------------------------------------------------

# model run-settings
# colocated run-settings

# model batch settings
# colocated batch settings


def test_launch_preview_orc_jp(fileutils, wlmutils):
    """just an orchestrator"""
    exp_name = "test_orc_active_functions_jp"
    exp = Experiment(exp_name, launcher="local")
    # without anything
    # print("\nwithout anytihng")
    # exp.preview()  # do we want it to work before with just the exp name and launcher?
    # correctly giving database status as inactive here

    # databases status -- not launching ..
    # if no entity .. path, launcher, status

    test_dir = fileutils.make_test_dir()

    db = Orchestrator(port=wlmutils.get_test_port())
    db.set_path(test_dir)

    # exp.preview(db)

    exp.start(db)
    exp.stop(db)


def test_launch_model_orc_jp(fileutils, wlmutils, mlutils):
    # regular model, regular run settings
    exp_name = "test_launch_model_orc_jp"
    exp = Experiment(exp_name, launcher="local")

    settings = exp.create_run_settings("echo", exe_args="Hello World")
    model = exp.create_model("hello_world", settings)

    exp.start(model, block=True, summary=True)

    print(exp.get_status(model))


# exp.stop(model) ?


def test_db_model_ensemble_jp(fileutils, wlmutils, mlutils):
    ## generate and start - the entities -- what about the time things aren't generated?
    # should I do a try: block?
    """Test DBModels on remote DB, with an ensemble"""

    # Set experiment name
    exp_name = "test-db-model-ensemble_jp"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create RunSettings
    run_settings = exp.create_run_settings("echo", exe_args="Hello World")
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # Create ensemble
    smartsim_ensemble = exp.create_ensemble(
        "smartsim_model", run_settings=run_settings, replicas=2
    )
    smartsim_ensemble.set_path(test_dir)

    # Create Model
    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    # Create database
    db = exp.create_database(port=test_port, interface=test_interface)
    exp.generate(db)

    # Add new ensemble member
    smartsim_ensemble.add_model(smartsim_model)

    # Launch and check successful completion
    try:
        exp.start(db, smartsim_ensemble, block=True)

    finally:
        exp.stop(db)


def test_runsettings_colo_jp(fileutils, wlmutils, mlutils):
    """Test DB Scripts on colocated DB"""

    # Set the experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    # test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    # Create RunSettings
    colo_settings = exp.create_run_settings("echo", exe_args="Hello World")
    # exe=sys.executable, exe_args=test_script)

    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create model with colocated database
    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db_tcp(
        port=test_port,
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    try:
        exp.start(colo_model, block=True, summary=True)
        # statuses = exp.get_status(colo_model)

    finally:
        exp.stop(colo_model)


########### now with batch settings
# salloc -N 3 -A account --exclusive -t 03:00:00
# export SMARTSIM_TEST_LAUNCHER=slurm


def test_batch_model_settings_jp():
    exp = Experiment("hello_world_batch", launcher="auto")

    # define resources for all ensemble members
    batch = exp.create_batch_settings(nodes=4, time="00:10:00", account="12345-Cray")

    batch.set_queue("premium")

    # define how each member should run
    run = exp.create_run_settings(exe="echo", exe_args="Hello World!")
    run.set_tasks(60)
    run.set_tasks_per_node(20)

    ensemble = exp.create_ensemble(
        "hello_world", batch_settings=batch, run_settings=run, replicas=4
    )
    exp.start(ensemble, block=True, summary=True)
    print(exp.get_status(ensemble))

    exp.stop(ensemble)


# batch colocated model  ?

# if hasattr(self.run_settings, "mpmd") and len(self.run_settings.mpmd) > 0:
#     raise SSUnsupportedError(
#         "Models colocated with databases cannot be run as a mpmd workload"
#     )Multiple-Program-Multiple-Data (MPMD) model

# why wouldnt colocated models work on wlm


# model status is paused forever
def test_batch_model_jp_1(fileutils, wlmutils):
    """Test the launch of a manually construced batch model"""

    exp_name = "test-batch-model"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")

    batch_settings = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch_settings.set_account(wlmutils.get_test_account())

    # if wlmutils.get_test_launcher() == "cobalt":
    #      batch_settings.set_queue("debug-flat-quad")

    run_settings = wlmutils.get_run_settings("python", f"{script} --time=5")

    # model with run settings and batch settings
    model = exp.create_model(
        "model", path=test_dir, run_settings=run_settings, batch_settings=batch_settings
    )
    model.set_path(test_dir)

    exp.start(model, block=True)

    exp.stop(model)  # jp added .. ?


# ?
# def test_batch_model_settings_jp_2(fileutils, wlmutils):
#     exp = Experiment("batch-db-on-pbs", launcher="auto")
#     # where batch is just true in create_database
#     db_cluster = exp.create_database(
#         db_nodes=3,
#         db_port=6780,
#         batch=True,
#         time="00:10:00",
#         interface="ib0",
#         account="12345-Cray",
#         queue="cl40",
#     )

#     exp.start(db_cluster)

#     print(f"Orchestrator launched on nodes: {db_cluster.hosts}")
#     # launch models, analysis, training, inference sessions, etc
#     # that communicate with the database using the SmartRedis clients

#     exp.stop(db_cluster)


# ?
def test_batch_model_settings_jp_3(fileutils, wlmutils):
    """model status is paused forever"""
    exp_name = "test-batch-ensemble-replicas"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")

    batch = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch.set_account(wlmutils.get_test_account())
    # if wlmutils.get_test_launcher() == "cobalt":
    #     # As Cobalt won't allow us to run two
    #     # jobs in the same debug queue, we need
    #     # to make sure the previous test's one is over
    #     sleep(30)
    #     batch.set_queue("debug-flat-quad")

    # ensemble with batch and run settings
    ensemble = exp.create_ensemble(
        "batch-ens-replicas", batch_settings=batch, run_settings=settings, replicas=2
    )
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)

    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_batchsettings_colo_jp(fileutils, wlmutils, mlutils):
    """Test DB Scripts on colocated DB"""

    # Set the experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    # test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    # Create RunSettings
    colo_settings = exp.create_run_settings("echo", exe_args="Hello World")
    # exe=sys.executable, exe_args=test_script)

    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create model with colocated database
    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db_tcp(
        port=test_port,
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    try:
        exp.start(colo_model, block=True)
        # statuses = exp.get_status(colo_model)

    finally:
        exp.stop(colo_model)


# copied from another test
def test_create_run_settings_input_mutation_jp():
    # Tests that the run args passed in are not modified after initialization
    key0, key1, key2 = "arg0", "arg1", "arg2"
    val0, val1, val2 = "val0", "val1", "val2"

    default_run_args = {
        key0: val0,
        key1: val1,
        key2: val2,
    }
    rs0 = create_run_settings(
        "local", "echo", "hello", run_command="auto", run_args=default_run_args
    )

    ## preview run settings

    # Confirm initial values are set
    assert rs0.run_args[key0] == val0
    assert rs0.run_args[key1] == val1
    assert rs0.run_args[key2] == val2

    # Update our common run arguments
    val2_upd = f"not-{val2}"
    default_run_args[key2] = val2_upd

    # Confirm previously created run settings are not changed
    assert rs0.run_args[key2] == val2


def test_ensemble_start_jp(fileutils):
    # Init Experiment and specify to launch locally
    exp = Experiment(name="Experiment-test-jp", launcher="local")
    rs = exp.create_run_settings(
        exe="python",
        exe_args="output_my_parameter_new_tag.py",  # /lus/cls01029/putko/test/smartsim/
    )

    # /lus/cls01029/putko/test/smartsim/tests/output_my_parameter_new_tag.py

    params = {"tutorial_name": ["Ellie", "John"], "tutorial_parameter": [2, 11]}
    ensemble = exp.create_ensemble(
        "ensemble_new_tag", params=params, run_settings=rs, perm_strategy="all_perm"
    )

    config_file = "./output_my_parameter_new_tag.py"
    ensemble.attach_generator_files(to_configure=config_file)

    exp.generate(ensemble, overwrite=True, tag="@")
    exp.start(ensemble, summary=True)


def test_batch_ensemble_jp(fileutils):
    # hello_ensemble.py
    # from smartsim import Experiment

    ## put a model on an ensembles???

    exp = Experiment("hello_world_batch", launcher="auto")

    # define resources for all ensemble members
    batch = exp.create_batch_settings(nodes=4, time="00:10:00", account="12345-Cray")
    # batch.set_queue("premium")

    # define how each member should run
    run = exp.create_run_settings(exe="echo", exe_args="Hello World!")
    run.set_tasks(60)
    run.set_tasks_per_node(20)

    ensemble = exp.create_ensemble(
        "hello_world", batch_settings=batch, run_settings=run, replicas=4
    )
    exp.start(ensemble, block=True, summary=True)

    print(exp.get_status(ensemble))


def test_another_batch_ensemble_jp(fileutils):
    import numpy as np
    from smartsim import Experiment

    exp = Experiment("Training-Run", launcher="slurm")

    # setup ensemble parameter space
    learning_rate = list(np.linspace(0.01, 0.5))
    train_params = {"LR": learning_rate}

    # define resources for all ensemble members
    sbatch = exp.create_batch_settings(
        nodes=4, time="01:00:00", account="12345-Cray", queue="gpu"
    )

    # define how each member should run
    srun = exp.create_run_settings(exe="python", exe_args="./train-model.py")
    srun.set_nodes(1)
    srun.set_tasks(24)

    ensemble = exp.create_ensemble(
        "Training-Ensemble",
        params=train_params,
        params_as_args=["LR"],
        batch_settings=sbatch,
        run_settings=srun,
        perm_strategy="random",
        n_models=4,
    )
    exp.start(ensemble, summary=True)


def test_key_prefixing_ensemble(fileutils):
    # calling ensemble.enable_key_prefixing() causes the SSKEYOUT environment variable to
    # be set,

    # if the model for the ensemble member has incoming entities (such as those set via
    # model.register_incoming_entity() or ensemble.register_incoming_entity()), the SSKEYIN environment
    # variable will be set

    # from smartsim import Experiment

    exp = Experiment("Training-Run", launcher="slurm")

    # setup ensemble parameter space
    learning_rate = list(np.linspace(0.01, 0.5))
    train_params = {"LR": learning_rate}

    # define resources for all ensemble members
    sbatch = exp.create_batch_settings(
        nodes=4, time="01:00:00", account="12345-Cray"
    )  # queue="gpu"

    # define how each member should run
    srun = exp.create_run_settings(exe="python", exe_args="./train-model.py")
    srun.set_nodes(1)
    srun.set_tasks(24)

    ensemble = exp.create_ensemble(
        "Training-Ensemble",
        params=train_params,
        params_as_args=["LR"],
        batch_settings=sbatch,
        run_settings=srun,
        perm_strategy="random",
        n_models=4,
    )

    # Enable key prefixing -- note that this should be done
    # before starting the experiment
    ensemble.enable_key_prefixing()

    exp.start(ensemble, summary=True)
