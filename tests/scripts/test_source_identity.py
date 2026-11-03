#!/usr/bin/env python3
"""Exercise source identities with small Git trees and non-computing runners."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[2]
HELPER = Path("tests/scripts/source_identity.py")


def run(command, env, code=0):
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == code, (command, result.returncode, result.stdout, result.stderr)
    return result.stdout.strip()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def init_repo(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root, "add", ".")
    git(root, "-c", "user.name=Identity Test", "-c", "user.email=identity@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "source fixture")
    return git(root, "rev-parse", "HEAD")


def populate(root):
    (root / HELPER).parent.mkdir(parents=True)
    shutil.copyfile(PROJECT / HELPER, root / HELPER)
    (root / "CMakeLists.txt").write_text("# identity fixture\n")


def identity(root, env, *args, code=0):
    return run([sys.executable, "-B", str(root / HELPER), *args], env, code)


def main():
    environment = os.environ.copy()
    environment.pop("LBM_EXPECTED_GIT_COMMIT", None)
    scratch = Path(os.environ.get("TMPDIR", Path(__file__).parent / "build"))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-identity-", dir=scratch) as work:
        work = Path(work)
        public = work / "public"
        populate(public)
        public_sha = init_repo(public)
        internal = work / "internal" / "LBMReX"
        populate(internal)
        internal_sha = init_repo(internal.parent)
        archive = work / "archive"
        populate(archive)
        unrelated = work / "unrelated"
        unrelated.mkdir()
        (unrelated / "unrelated.txt").write_text("unrelated history\n")
        unrelated_sha = init_repo(unrelated)
        nested_archive = unrelated / "LBMReX"
        populate(nested_archive)

        for root, expected in ((public, public_sha), (internal, internal_sha)):
            assert identity(root, environment, "--strict") == expected
            assert identity(root, {**environment, "LBM_EXPECTED_GIT_COMMIT": expected},
                            "--expected", expected, "--strict") == expected
            identity(root, {**environment, "LBM_EXPECTED_GIT_COMMIT": "b" * 40},
                     "--strict", code=2)
        for root in (archive, nested_archive):
            assert identity(root, environment) == "unknown"
            identity(root, environment, "--strict", code=2)
            assert identity(root, {**environment, "LBM_EXPECTED_GIT_COMMIT": public_sha},
                            "--strict") == public_sha
            identity(root, {**environment, "LBM_EXPECTED_GIT_COMMIT": "unknown"},
                     "--strict", code=2)
            identity(root, {**environment, "LBM_EXPECTED_GIT_COMMIT": public_sha},
                     "--expected", "b" * 40, code=2)
        assert identity(nested_archive, environment) != unrelated_sha
        no_git = {**environment, "PATH": "", "LBM_EXPECTED_GIT_COMMIT": public_sha}
        assert identity(archive, no_git, "--strict") == public_sha
        print("PASS: internal/public clone, archive, unrelated parent, missing/invalid/mismatched identity, no Git")

        # Run the production CMake module in script mode, without configuring AMReX/LBM.
        cmake_script = work / "identity.cmake"
        generated = work / "PerformanceBuildInfo.H"
        cmake_script.write_text(
            'include("' + str(PROJECT / "cmake/SourceIdentity.cmake") + '")\n'
            'set(LBM_PERFORMANCE_GIT_COMMIT "' + '$' + '{LBM_SOURCE_ID}")\n'
            'configure_file("' + str(PROJECT / "tests/performance/src/PerformanceBuildInfo.H.in")
            + '" "' + str(generated) + '" @ONLY)\n'
            'file(WRITE "' + str(work / "environment.txt") + '" "' + '$' + '{LBM_SOURCE_ENVIRONMENT}")\n'
        )
        for root, expected in ((public, public_sha), (internal, internal_sha),
                               (archive, public_sha), (nested_archive, public_sha)):
            run(["cmake", "-DPROJECT_SOURCE_DIR=" + str(root),
                 "-DPython3_EXECUTABLE=" + sys.executable,
                 "-DLBM_EXPECTED_GIT_COMMIT=" + expected, "-P", str(cmake_script)], environment)
            assert expected in generated.read_text()
            assert (work / "environment.txt").read_text() == "LBM_EXPECTED_GIT_COMMIT=" + expected
        run(["cmake", "-DPROJECT_SOURCE_DIR=" + str(public),
             "-DPython3_EXECUTABLE=" + sys.executable,
             "-DLBM_EXPECTED_GIT_COMMIT=" + "b" * 40, "-P", str(cmake_script)], environment, 1)
        print("PASS: CMake generated build identity and CTest environment, mismatched clone rejection")

        # The launcher records real runner argv then fails before any solver/checker executes.
        launcher = work / "launcher"
        launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$IDENTITY_ARGV"\nexit 7\n')
        launcher.chmod(0o755)
        entries = {
            "amr_channel_cylinder/run.sh": "d2-yms-fine",
            "amr_channel_cylinder/run_dynamic.sh": "d2-single",
            "amr_channel_cylinder/run_re100.sh": "d2-yms-fine",
            "amr_channel_cylinder/run_tandem.sh": "d2-yms-cross-distinct",
            "amr_channel_cylinder/run_tandem_compare.sh": "same",
            "amr_double_shear_layer/run.sh": "d2-ref128",
            "amr_double_shear_layer/run_dynamic.sh": "d2-2l",
            "amr_lid_driven_cavity/run.sh": "d2-fine",
            "amr_periodic_shear/run.sh": "d2-axis",
            "amr_taylor_green_vortex/run.sh": "d2-20-uniform",
            "amr_uniform_flow/run.sh": "d2",
        }
        argv_path = work / "argv.txt"
        run_env = {**environment, "LBM_EXPECTED_GIT_COMMIT": public_sha,
                   "LBM_TEST_PYTHON": sys.executable, "LBM_TEST_MAX_STEP": "1",
                   "LBM_TEST_AMR_MPI_RANKS": "1", "IDENTITY_ARGV": str(argv_path)}
        for index, (relative, entry) in enumerate(entries.items()):
            runner = archive / "tests/numerical" / relative
            runner.parent.mkdir(parents=True, exist_ok=True)
            original = PROJECT / "tests/numerical" / relative
            shutil.copyfile(original, runner)
            inputs = original.parent / "inputs"
            if inputs.exists() and not (runner.parent / "inputs").exists():
                shutil.copytree(inputs, runner.parent / "inputs")
            output = work / ("run-" + str(index))
            if relative.endswith("run_tandem_compare.sh"):
                args = ["unused-exe", str(output), str(launcher), entry]
            else:
                args = ["unused-d2", "unused-d3", str(output), str(launcher), entry]
            if argv_path.exists():
                argv_path.unlink()
            run(["bash", str(runner), *args], run_env, 2)
            assert argv_path.exists(), relative
            assert "run.source_id=" + public_sha in argv_path.read_text().splitlines(), relative
            if relative == "amr_channel_cylinder/run.sh":
                records = list(output.glob("attempt-*/result.txt"))
                assert len(records) == 1
                assert "source_id=" + public_sha in records[0].read_text().splitlines()
            argv_path.unlink()
            run(["bash", str(runner), *args], environment, 2)
            assert not argv_path.exists(), relative
        print("PASS: all 11 numerical runners propagate the archive SHA, Re20 result records the SHA, and missing identity is rejected before execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
