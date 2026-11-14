"""Stage a Cygwin/MSVC static PETSc build for native Windows petsc4py.

Run with Windows Python, passing the PETSc source directory, PETSC_ARCH, and
destination prefix. The original Cygwin build configuration stays intact.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("arch")
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    build = source / args.arch
    prefix = args.prefix.resolve()
    if not (build / "lib/libpetsc.lib").is_file():
        raise FileNotFoundError("Build PETSc's MSVC static library before staging it.")
    sys.path.insert(0, str(source / "src/binding/petsc4py/conf"))
    from confpetsc import PetscConfig, split_quoted

    config = PetscConfig(source.as_posix(), args.arch)
    if config["BUILDSHAREDLIB"] == "yes":
        raise ValueError("This staging helper requires a static PETSc build.")

    shutil.copytree(source / "include", prefix / "include", dirs_exist_ok=True)
    shutil.copytree(build / "include", prefix / "include", dirs_exist_ok=True)
    library_dir = prefix / "lib"
    library_dir.mkdir(parents=True, exist_ok=True)
    libraries = list((build / "lib").glob("*.lib"))
    if not libraries:
        raise FileNotFoundError("No MSVC libraries found in the PETSc build.")
    for library in libraries:
        shutil.copy2(library, library_dir / library.name)

    def native_paths(value: str) -> str:
        value = re.sub(
            r"/cygdrive/([a-zA-Z])/", lambda match: match[1].upper() + ":/", value
        )
        value = value.replace(build.as_posix(), prefix.as_posix())
        return value.replace(source.as_posix(), prefix.as_posix())

    def native_library(name: str) -> str:
        for filename in (f"lib{name}.lib", f"{name}.lib"):
            if (library_dir / filename).is_file():
                return (library_dir / filename).as_posix()
        return f"{name}.lib"

    external_flags = []
    for flag in split_quoted(native_paths(config["PETSC_EXTERNAL_LIB_BASIC"])):
        if flag.startswith("-L"):
            flag = "/LIBPATH:" + flag[2:]
        elif flag.startswith("-l"):
            flag = native_library(flag[2:])
        external_flags.append(flag)

    conf_dir = library_dir / "petsc/conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "lib/petsc/conf/variables", conf_dir / "variables")
    variables = (build / "lib/petsc/conf/petscvariables").read_text()
    overrides = {
        "PETSC_DIR": prefix.as_posix(),
        "PETSC_ARCH": "",
        "PETSC_LIB_DIR": library_dir.as_posix(),
        "PETSC_LIB_BASIC": "-llibpetsc",
        "PETSC_CC_INCLUDES": native_paths(config["PETSC_CC_INCLUDES"]),
        "PETSC_EXTERNAL_LIB_BASIC": " ".join(external_flags),
    }
    variables += "\n# Native Windows Python binding configuration\n"
    variables += "".join(f"{key} = {value}\n" for key, value in overrides.items())
    (conf_dir / "petscvariables").write_text(variables)
    print(f"Staged {len(libraries)} MSVC libraries in {prefix}")


if __name__ == "__main__":
    main()
