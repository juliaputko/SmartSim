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

import pytest

from smartsim import Experiment
from smartsim._core import Manifest
from smartsim._core import previewrenderer
import pathlib


def test_experiment_preview(test_dir, wlmutils):
    """Test correct preview output items for Experiment preview"""
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_prefix"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # Call method for string formatting for testing
    output = previewrenderer.render(exp)
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "").strip() for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert set(["Experiment", "Experiment Path", "Launcher"]).issubset(summary_dict)


def test_experiment_preview_properties(test_dir, wlmutils):
    """Test correct preview output properties for Experiment preview"""
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_experiment_preview_properties"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    # Call method for string formatting for testing
    output = previewrenderer.render(exp)
    summary_lines = output.split("\n")
    summary_lines = [item.replace("\t", "").strip() for item in summary_lines[-3:]]
    assert 3 == len(summary_lines)
    summary_dict = dict(row.split(": ") for row in summary_lines)
    assert exp.name == summary_dict["Experiment"]
    assert exp.exp_path == summary_dict["Experiment Path"]
    assert exp.launcher == summary_dict["Launcher"]


def test_model_jp(test_dir, wlmutils):
    exp_name = "test_model_preview"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)
    rendered_preview = previewrenderer.render(exp, preview_manifest)
    print(rendered_preview)


def test_model_preview(test_dir, wlmutils):
    exp_name = "test_model_preview"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)
    rendered_preview = previewrenderer.render(exp, preview_manifest)
    assert "Model name" in rendered_preview
    assert "Executable" in rendered_preview
    assert "Executable Arguments" in rendered_preview


def test_model_preview_parameters(test_dir, wlmutils):
    exp_name = "test_model_preview"
    test_launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)

    preview_manifest = Manifest(hello_world_model, spam_eggs_model)
    rendered_preview = previewrenderer.render(exp, preview_manifest)

    # for hello world model
    assert "echo-hello" in rendered_preview
    assert "/usr/bin/echo" in rendered_preview
    assert "hello" in rendered_preview
    assert "world" in rendered_preview
    assert "echo-hello" == hello_world_model.name
    assert "/usr/bin/echo" == hello_world_model.run_settings.exe[0]
    assert "hello" == hello_world_model.run_settings.exe_args[0]
    assert "world" == hello_world_model.run_settings.exe_args[1]
    # for spam eggs model
    assert "echo-spam" in rendered_preview
    assert "/usr/bin/echo" in rendered_preview
    assert "spam" in rendered_preview
    assert "eggs" in rendered_preview
    assert "echo-spam" == spam_eggs_model.name
    assert "/usr/bin/echo" == spam_eggs_model.run_settings.exe[0]
    assert "spam" == spam_eggs_model.run_settings.exe_args[0]
    assert "eggs" == spam_eggs_model.run_settings.exe_args[1]


def test_preview_output_format_html(test_dir, wlmutils):
    """Test correct preview output items for Experiment preview"""
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_prefix"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    filename = "preview_test.html"
    path = pathlib.Path() / "preview_test.html"
    # call preview with output format and output filename
    exp.preview(output_format="html", output_filename=filename)
    assert path.exists()
    assert path.is_file()


def test_output_filename_without_format(test_dir, wlmutils):
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_prefix"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    filename = "preview_test.html"
    # call preview with output filename
    with pytest.raises(ValueError) as ex:
        exp.preview(output_filename=filename)
    assert (
        "Output filename is only a valid parameter when an output format is specified"
        in ex.value.args[0]
    )


def test_output_format_without_filename(test_dir, wlmutils):
    test_launcher = wlmutils.get_test_launcher()
    exp_name = "test_prefix"
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)
    filename = "preview_test.html"
    # call preview with output filename
    with pytest.raises(ValueError) as ex:
        exp.preview(output_format="html")
    assert (
        "An output filename is required when an output format is set."
        in ex.value.args[0]
    )


def test_output_format_error():
    exp_name = "test_output_format"
    exp = Experiment(exp_name)
    with pytest.raises(ValueError) as ex:
        exp.preview(output_format="hello")
    assert "The only valid currently available is html" in ex.value.args[0]


def test_verbosity_level_type_error():
    exp_name = "test_verbosity_level"
    exp = Experiment(exp_name)

    with pytest.raises(ValueError) as ex:
        exp.preview(verbosity_level="hello")
    assert (
        "The only valid verbosity level currently available is info" in ex.value.args[0]
    )


def test_verbosity_level_debug_error():
    exp_name = "test_output_format"
    exp = Experiment(exp_name)
    with pytest.raises(NotImplementedError):
        exp.preview(verbosity_level="debug")


def test_verbosity_level_developer_error():
    exp_name = "test_output_format"
    exp = Experiment(exp_name)
    with pytest.raises(NotImplementedError):
        exp.preview(verbosity_level="developer")
