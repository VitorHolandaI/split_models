## Code style

- Functions: 4-20 lines. Split if longer.
- Files: under 500 lines. Split by responsibility.
- One thing per function, one responsibility per module (SRP).
- Names: specific and unique. Avoid `data`, `handler`, `Manager`.
  Prefer names that return <5 grep hits in the codebase.
- Types: explicit. No `any`, no `Dict`, no untyped functions.
- No code duplication. Extract shared logic into a function/module.
- Early returns over nested ifs. Max 2 levels of indentation.
- Exception messages must include the offending value and expected shape.
- Since we are using Exceptions we need to always use them when making requests, since its a way to capture a error tha may occour
- Having silence erros its bad pratice.

## Comments

- Keep your own comments. Don't strip them on refactor — they carry
  intent and provenance.
- Write WHY, not WHAT. Skip `// increment counter` above `i++`.
- Docstrings on public functions: intent + one usage example.
- Reference issue numbers / commit SHAs when a line exists because
  of a specific bug or upstream constraint.

## Tests

- Tests run with a single command: `<project-specific>`.
- Every new function gets a test. Bug fixes get a regression test.
- Mock external I/O (API, DB, filesystem) with named fake classes,
  not inline stubs.
- Tests must be F.I.R.S.T: fast, independent, repeatable,
  self-validating, timely.

## Dependencies

- Inject dependencies through constructor/parameter, not global/import.
- Wrap third-party libs behind a thin interface owned by this project.

## Structure

- Follow the framework's convention (Rails, Django, Next.js, etc.).
- Prefer small focused modules over god files.
- Predictable paths: controller/model/view, src/lib/test, etc.

## Formatting

- Use the language default formatter (`cargo fmt`, `gofmt`, `prettier`,
  `black`, `rubocop -A`). Don't discuss style beyond that.

## Logging

- Structured JSON when logging for debugging / observability.
- Plain text only for user-facing CLI output.

## memory
- At the start of non-trivial tasks, call `memory_smart_search` with the task
- keywords. Use `memory_save` or `memory_lesson_save` when capturing decisions,
- patterns, or preferences worth keeping.
- Always avery increat of 10k context lenght save to ai-memory
- every 100k of contexto call ai-memory tool to save

## Author
- The autor is  the User dont use made with... etc etc etc

## TDD
- Use TDD pois sempre quando tiver mudancas drasticas no codigo ou pergute ao usuario se quer que rode os testes apos a mudanca é o hardness que falta
- Use Tambem a depender do tipo de codigo front ou back cobertura de testes é importante para manter estavel o codigo
- Use https://github.com/VitorHolandaI/quick_python_analisys.git
