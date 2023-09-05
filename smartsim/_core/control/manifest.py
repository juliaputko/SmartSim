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

import typing as t

from ...database import Orchestrator
from ...entity import EntityList, SmartSimEntity, Model, Ensemble
from ...error import SmartSimError
from ..utils.helpers import fmt_dict


class Manifest:
    """This class is used to keep track of all deployables generated by an experiment.
    Different types of deployables (i.e. different `SmartSimEntity`-derived objects or
    `EntityList`-derived objects) can be accessed by using the corresponding accessor.

    Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
    can all be passed as arguments
    """

    def __init__(self, *args: SmartSimEntity) -> None:
        self._deployables = list(args)
        self._check_types(self._deployables)
        self._check_names(self._deployables)
        self._check_entity_lists_nonempty()

    @property
    def dbs(self) -> t.Optional[list]:
        """Return a list of Orchestrator instances in Manifest

        :raises SmartSimError: if user added to databases to manifest
        :return: List of orchestrator instances
        :rtype: list[Orchestrator] | None
        """
        dbs = [item for item in self._deployables if isinstance(item, Orchestrator)]
        return dbs if dbs else None

    @property
    def models(self) -> t.List[Model]:
        """Return Model instances in Manifest

        :return: model instances
        :rtype: List[Model]
        """
        _models: t.List[Model] = [
            item for item in self._deployables if isinstance(item, Model)
        ]
        return _models

    @property
    def ensembles(self) -> t.List[Ensemble]:
        """Return Ensemble instances in Manifest

        :return: list of ensembles
        :rtype: List[Ensemble]
        """
        return [e for e in self._deployables if isinstance(e, Ensemble)]

    @property
    def all_entity_lists(self) -> t.List[EntityList]:
        """All entity lists, including ensembles and
        exceptional ones like Orchestrator

        :return: list of entity lists
        :rtype: List[EntityList]
        """
        _all_entity_lists: t.List[EntityList] = []
        _all_entity_lists.extend(self.ensembles)

        dbs = self.dbs

        if dbs is not None:
            for a_db in dbs:
                _all_entity_lists.append(a_db)

        return _all_entity_lists

    @staticmethod
    def _check_names(deployables: t.List[t.Any]) -> None:
        used = []
        for deployable in deployables:
            name = getattr(deployable, "name", None)
            if not name:
                raise AttributeError("Entity has no name. Please set name attribute.")
            if name in used:
                raise SmartSimError("User provided two entities with the same name")
            used.append(name)

    @staticmethod
    def _check_types(deployables: t.List[t.Any]) -> None:
        for deployable in deployables:
            if not isinstance(deployable, (SmartSimEntity, EntityList)):
                raise TypeError(
                    f"Entity has type {type(deployable)}, not "
                    + "SmartSimEntity or EntityList"
                )

    def _check_entity_lists_nonempty(self) -> None:
        """Check deployables for sanity before launching"""

        for entity_list in self.all_entity_lists:
            if len(entity_list) < 1:
                raise ValueError(f"{entity_list.name} is empty. Nothing to launch.")

    def __str__(self) -> str:
        output = ""
        e_header = "=== Ensembles ===\n"
        m_header = "=== Models ===\n"
        db_header = "=== Database ===\n"
        if self.ensembles:
            output += e_header

            all_ensembles = self.ensembles
            for ensemble in all_ensembles:
                output += f"{ensemble.name}\n"
                output += f"Members: {len(ensemble)}\n"
                output += f"Batch Launch: {ensemble.batch}\n"
                if ensemble.batch:
                    output += f"{str(ensemble.batch_settings)}\n"
            output += "\n"

        if self.models:
            output += m_header
            for model in self.models:
                output += f"{model.name}\n"
                if model.batch_settings:
                    output += f"{model.batch_settings}\n"
                output += f"{model.run_settings}\n"
                if model.params:
                    output += f"Parameters: \n{fmt_dict(model.params)}\n"
            output += "\n"

        if self.dbs:
            for adb in self.dbs:
                output += db_header
                output += f"Shards: {adb.num_shards}\n"
                output += f"Port: {str(adb.ports[0])}\n"
                output += f"Network: {adb._interfaces}\n"
                output += f"Batch Launch: {adb.batch}\n"
                if adb.batch:
                    output += f"{str(adb.batch_settings)}\n"

        output += "\n"
        return output

    @property
    def has_db_objects(self) -> bool:
        """Check if any entity has DBObjects to set"""

        def has_db_models(entity: t.Union[EntityList, Model]) -> bool:
            return len(list(entity.db_models)) > 0

        def has_db_scripts(entity: t.Union[EntityList, Model]) -> bool:
            return len(list(entity.db_scripts)) > 0

        has_db_objects = False
        for model in self.models:
            has_db_objects |= hasattr(model, "_db_models")

        # Check if any model has either a DBModel or a DBScript
        # we update has_db_objects so that as soon as one check
        # returns True, we can exit
        has_db_objects |= any(
            has_db_models(model) | has_db_scripts(model) for model in self.models
        )
        if has_db_objects:
            return True

        # If there are no ensembles, there can be no outstanding model
        # to check for DBObjects, return current value of DBObjects, which
        # should be False
        ensembles = self.ensembles
        if not ensembles:
            return has_db_objects

        # First check if there is any ensemble DBObject, if so, return True
        has_db_objects |= any(
            has_db_models(ensemble) | has_db_scripts(ensemble) for ensemble in ensembles
        )
        if has_db_objects:
            return True
        for ensemble in ensembles:
            # Last case, check if any model within an ensemble has DBObjects attached
            has_db_objects |= any(
                has_db_models(model) | has_db_scripts(model)
                for model in ensemble.models
            )
            if has_db_objects:
                return True

        # `has_db_objects` should be False here
        return has_db_objects
