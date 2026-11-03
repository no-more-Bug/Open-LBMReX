# Contributing

Open-LBMReX is a personal hobby codebase and an exercise in AI-assisted coding. Issues and focused pull requests are welcome, but there is no guaranteed response time, maintenance schedule or correctness assurance.

## Propose a focused change

For changes to numerical methods, AMR, boundaries, parallel execution or output formats, discuss the intended behavior and validation first. Explain the problem, the smallest useful change and any effect on existing results. Keep unrelated refactoring and large generated changes separate.

Use the public commit or release tag when reporting a problem. Include a small reproducer, inputs, relevant commands and the expected versus observed behavior. Remove personal paths, credentials, account details and unrelated logs before sharing material.

## Source ownership and snapshots

Development continues in a private repository, and this repository receives curated source snapshots. Changes to source, tests, examples and managed documentation are brought back into the internal source before the next snapshot, with contributor authorship and attribution preserved. The maintainer coordinates this transfer and the public merge; internal Git history is not merged into the public repository.

A fix accepted directly in the public repository must also be carried back internally before the next export, so that a later snapshot does not overwrite it. CONTRIBUTING.md, CHANGELOG.md and any future .github content are maintained in the public repository. Contributors do not need access to the private repository.

Use the author name you want published; a pseudonym and a GitHub noreply address are welcome. Keep existing copyright and attribution notices. Do not replace another contributor's authorship with the maintainer's identity.

## Validation

Start with the [build instructions](docs/building.md) and [test documentation](tests/README.md). Run checks appropriate to the change and report exactly what ran, including the source revision, AMReX identity and patch, compiler, backend and relevant MPI/thread/device configuration. Clearly identify tests that failed or were not run.

A successful build or short engineering test does not establish scientific validity. Numerical changes need independent expectations or references and must preserve the distinction between passed scientific criteria, failed criteria and insufficient evidence. Read the [validation scope](docs/validation.md) and [known limitations](docs/known-limitations.md).

AI-assisted contributions are welcome. Explain material use of AI tools and check the proposed changes and their sources; do not treat generated code or text as automatically correct.

## License and third-party material

Submit only material you have the right to contribute under the project's [BSD-3-Clause license](LICENSE), retaining any applicable copyright and attribution. Third-party code, data and figures must retain their own terms and provenance; the project license does not replace those terms. Include source and permission information for new third-party material. See [third-party notices](docs/third-party.md). No additional copyright-transfer agreement is requested.
