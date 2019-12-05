import pickle
import sys
from os import path, mkdir, listdir, getcwd
from .error import SmartSimError, SSConfigError
from .target import Target
from .model import NumModel
from .orchestrator import Orchestrator
from .smartSimNode import SmartSimNode

from .utils import get_logger
logger = get_logger(__name__)


class State:
    """The State class holds most of the configurations necessary to run an
       experiment within SmartSim. State is responsible for the creation
       and storage of SmartSim entities such as models that are used to
       construct data pipelines.

       :param str experiment: Name of the directory that will house all of
                              the created files and directories.
    """

    def __init__(self, experiment):
        self.current_state = "Initializing"
        self.experiment = experiment
        self.targets = []
        self.nodes = []
        self.orc = None


    def __str__(self):
        state_str = "\n-- State Summary --\n"
        if len(self.targets) > 0:
            state_str += "\n-- Targets --"
            for target in self.targets:
                state_str += str(target)
        if len(self.nodes) > 0:
            state_str += "\n-- Nodes --"
            for node in self.nodes:
                state_str += str(node)
        if self.orc:
            state_str += str(self.orc)
        return state_str


    def load_target(self, name, target_path=None):
        """Load a pickled target into State for use. The target currently must be from
           the same experiment it originated in. This can be useful if the experiment
           is being conducted over mutliple stages where execution does not all occur
           within the same script.

           :param str name: name of the pickled target
           :param str target_path: Path to the pickled target. Defaults to os.getcwd()

        """
        try:
            tar_dir = path.join(getcwd(), self.experiment, name)
            if target_path:
                tar_dir = target_path
            if path.isdir(tar_dir):
                pickle_file = path.join(tar_dir, name + ".pickle")
                if path.isfile(pickle_file):
                    target = pickle.load(open(pickle_file, "rb"))
                    if target.experiment != self.experiment:
                        err = "Target must be loaded from same experiment \n"
                        msg = "Target experiment: {}   Current experiment: {}".format(target.experiment,
                                                                                       self.experiment)
                        raise SmartSimError(err+msg)
                    self.targets.append(target)
                else:
                    raise SmartSimError("Target, {}, could not be found".format(name))
            else:
                raise SmartSimError("Target directory could not be found!")
        except SmartSimError as e:
            logger.error(e)
            raise


    def create_target(self, name, params={}, run_settings=None):
        """Create a target to be used within one or many of the SmartSim Modules. Targets
           keep track of groups of models. Parameters can be given to a target as well in
           order to generate models based on a combination of parameters and generation
           stategies. For more on generation strategies, see the Generator Class.

           :param str name: name of the new target
           :param dict params: dictionary of model parameters to generate models from based
                               on a run strategy.

        """
        try:
            for target in self.targets:
                if target.name == name:
                    raise SmartSimError("A target named " + target.name + " already exists!")

            target_path = path.join(getcwd(), self.experiment, name)
            if path.isdir(target_path):
                raise SmartSimError("Target directory already exists: " + target_path)
            new_target = Target(name, params, self.experiment, target_path, run_settings)
            self.targets.append(new_target)
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_model(self, name, target="default_target", params={}, path=None):
        """Create a model belonging to a specific target. This function is
           useful for running a small number of models where the model files
           are already in place for execution.

           If the target specified by the argument `target` doesn't exist,
           a new target is created and added to the state before the model
           is added to the target.  Calls to this function without specifying
           the `target` argument result in the creation/usage a target named
           "default_target", the default argument for `target`.

           :param str name: name of the model to be created
           :param str target: name of the target to place model into
           :param dict params: dictionary of model parameters
           :param str path: (optional) path to model files, defaults to os.getcwd()
        """
        model_added = False
        target_exists = False
        if not path:
            path = getcwd()
        for t in self.targets:
            if t.name == target:
                target_exists = True
                model = NumModel(name, params, path)
                t.add_model(model)
                model_added = True
        if not target_exists:
            # create a new target with name target.  Since create_target appends, we
            # pull the final target in self.targets
            self.create_target(name=target)
            model = NumModel(name, params, path)
            self.targets[-1].add_model(model)
            model_added = True
        if not model_added:
            raise SmartSimError("Could not find target by the name of: " + target)

    def create_orchestrator(self, name=None, port=6379, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        """Create an orchestrator database to faciliate the transfer of data
           for online training and inference. After the orchestrator is created,
           connections between models and nodes can be instantiated through a
           call to State.register_connection().

           :param str name: name of orchestrator, defaults to "Orchestrator"
           :param int port: the port to open database communications on
           :param int nodes: number of nodes to distribute the database over
                             (currently only works on 1 node)
           :param str duration: how long the orchestrator should run for.
                                format: H:M:S  e.g. "1:00:00"
           """
        settings = kwargs
        settings["nodes"] = nodes
        settings["ppn"] = ppn
        settings["duration"] = duration
        self.orc = Orchestrator(name=name, port=port, **settings)

    def create_node(self, name, script_path=None, **kwargs):
        """Create a SmartSimNode for a specific task. Examples of SmartSimNode
           tasks include training, processing, and inference. Nodes can be used
           to run any task written in any language. The included script/executable
           for nodes often use the Client class to send and recieve data from
           the SmartSim orchestrator.

           :param str name: name of the node to be launched
           :param str script_path: path to the script or executable to be launched.
                                   (default is the current working directory of the
                                    SmartSim run script)
           :param dict kwargs: Extra settings for the workload manager can be set by
                               inluding keyword arguments such as duration="1:00:00"
                               or nodes=5
           """
        # TODO get settings from config file as well
        node = SmartSimNode(name, path=script_path, **kwargs)
        self.nodes.append(node)

    def register_connection(self, sender, reciever):
        """Create a runtime connection in orchestrator for data to be passed between two
           SmartSim entities. The possible types of connections right now are:

                Model -> Node
                Node  -> Node
                Node  -> Model

           :param str sender: name of the created entity with a Client instance to send
                              data to a registered counterpart
           :param str reciever: name of the created entity that will recieve data by
                                making calls to a Client instance.
        """
        if not self.orc:
            raise SmartSimError("Create orchestrator to register connections")
        else:
            # TODO check if sender and reciever are registered entities
            # TODO check for illegal connection types. e.g. model to model
            self.orc.junction.register(sender, reciever)

    def delete_target(self, name):
        """Delete a created target from State so that any future calls to SmartSim
           Modules will not include this target.

           :param str name: name of the target to be deleted
           :raises SmartSimError: if target doesnt exist
        """
        # TODO delete the files as well if generated
        target_deleted = False
        for t in self.targets:
            if t.name == name:
                self.targets.remove(t)
                target_deleted = True
        if not target_deleted:
            raise SmartSimError("Could not delete target: " + name)

    def save(self):
        """Save each target currently in state as a pickled python object.
           All models within the target are maintained. Targets can be reloaded
           into an experiment through a call to state.load_target.
        """
        for target in self.targets:
            pickle_path = path.join(target.path, target.name + ".pickle")
            if not path.isdir(path.join(target.path, target.name)):
                raise SmartSimError("Targets must be generated in order to save them")
            file_obj = open(pickle_path, "wb")
            pickle.dump(target, file_obj)
            file_obj.close()

    def get_model(self, model, target):
        """Get a specific model from a target.

           :param str model: name of the model to return
           :param str target: name of the target where the model is located

           :returns: NumModel instance
        """
        try:
            target = self.get_target(target)
            model = target[model]
            return model
        # if the target is not found
        except SmartSimError:
            raise
        except KeyError:
            raise SmartSimError("Model not found: " + model)

    def get_target(self, target):
        """Return a specific target from State

           :param str target: Name of the target to return

           :returns: Target instance
           :raises: SmartSimError
        """
        for t in self.targets:
            if t.name == target:
                return t
        raise SmartSimError("Target not found: " + target)


    def get_expr_path(self):
        expr_path = path.join(getcwd(), self.experiment)
        return expr_path
