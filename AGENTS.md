# Coding standards

## Writing style

Applies to code, docs, and comments.

- No AI-slop prose: no em dashes, no dash-interjections (" - ") used as
  punctuation, no semicolons anywhere, never use the word "own", no "not X, but Y" constructions.
- No meta-commentary. State facts. Don't narrate that a convention was
  followed, and don't explain why a sentence exists.
- No dramatic intensifiers or superlatives ("directly", "straight to",
  "SOTA", "leader", "best-in-class"). State the plain mechanism instead.
- No emphatic or repetitive chat language. No exclamation-driven
  excitement, no repeating a word or phrase for emphasis.
- Lead with the substantive fact, not a minor or structural detail.

## Code and Tests

- Every part of the code must be vectorized, parallelized, and
  optimized, as a default, not an afterthought considered only after
  something runs slowly. Before writing or running any loop, ask
  whether its iterations are independent (parallelize across workers)
  and whether it accumulates into an array (vectorize with numpy). The
  two rules below are instances of this default, not its full scope.
- Code must be production ready, clean code, thoroughly tested by TDD.
- Validate that all tests are correct, complete, needed, and not vacuous.
- Every source file has real tests, written test-first.
- Tests follow FIRST: Fast, Independent, Repeatable, Self-validating,
  Timely.
- Never defer a lint, type-check, or test failure as "pre-existing."
  When a check is run, every failure it reports must be fixed in the
  same pass, not just the ones touching files already being edited.
  Tests and linting must always pass, full stop.
- Zero monkeypatches anywhere. Use dependency injection instead: every
  external dependency (HTTP clients, model loaders, torch/cuda checks,
  env vars, subprocess calls) is an explicit parameter that defaults to
  the real implementation, so a test substitutes a fake directly.
- Vectorize every numeric accumulation, with no exception for small
  inputs: replace a Python-level loop that mutates a numpy array once
  per iteration (e.g. `counts[i] += 1.0` per element) with a single
  batched array operation (e.g. `np.bincount`), regardless of whether
  the loop is a measured hot path. Looping over plain Python/dict
  objects to build the inputs to that one batched call (e.g. mapping
  strings to integer indices) is not itself a violation. Every such
  optimization must be lossless, computing the exact same statistic via
  the same formula, never an approximation or a lower-precision dtype.
- Any batch job scoring or processing many independent files or models
  (benchmark runs, dataset generation) must run across parallel workers,
  e.g. `ProcessPoolExecutor`, never as a single sequential per-item loop.
  Before invoking a script that loops sequentially over models, wrap it
  or replace it with a parallel driver calling the identical underlying
  functions, so results stay numerically identical while wall-clock time
  drops. This applies even to official or frozen scripts: freezing the
  scored numbers does not freeze the loop that produces them.

## Docstrings and comments

- Every comment and every docstring is one physical line, always, with
  zero exceptions. There is no "genuinely necessary context" exception.
  Delete content that doesn't fit rather than wrapping it to a second
  line.
- Zero extraneous comments. Write one only when the *why* is
  non-obvious. Never explain *what* the code does, identifiers already
  do that.
- Every class and method has a clean, concise one-line docstring.

## Citations and factual claims

- Verify any citation against the real primary source (fetch the actual
  page) before writing it down. Don't trust a search snippet or model
  memory.
- Citations live in documentation only. Never bake them into code,
  comments, or generated data.

## Statistical and research methodology

- Every metric, test, grouping, or correction procedure must be an
  established technique from the literature, verified against its
  primary source, not invented on the spot to fit the current data or
  a hoped-for result. Advanced, sophisticated methods are fine.
  Speculative or self-designed ones are not, no matter how principled
  they sound in the moment.
- Choose the specific established method deliberately for the
  problem's actual structure (sample size, class balance, dependence
  structure, data type), rather than reaching for whatever is most
  familiar. State the choice and its rationale plainly, don't leave
  the reader to infer why one method was picked over another.
- If no established method fits, say so explicitly and ask rather than
  filling the gap with an invented one.

## Architecture

- No feature flags or backwards-compatibility shims. Change the code
  directly.

## Working in these repos

- Never run git commands, under any circumstances. You may run read-only
  git commands ONLY (`git status`, `git log`, `git diff`).
- Don't run destructive operations without confirmation.
- Always open local previews/dev servers in real Chrome (Codex-in-chrome
  tools), never the built-in Browser pane.

## Research README standards

Apply these rules to the root README.

### Structure

- Use this heading order: `Overview`, `Data`, `Methodology`, `Results`,
  `Limitations`, `Reproducibility`, `Installation`, `Usage`, `References`,
  `License`.
- Give every section repository-specific content. Where a category is absent,
  state that fact and its consequence. Never leave a heading empty.

### Research account

- Begin `Overview` with a simple factual statement of what the repository does.
  Keep data description in `Data` and methodological argument in `Methodology`.
- In `Data`, define the materials, unit of analysis, origin, access conditions,
  transformations, and quality limits. State how the materials became data for
  the project. Annotations, metadata, and database fields encode decisions.
- In `Methodology`, present observable form and its registration before any
  interpretive claim. State mappings, measurements, exclusions, decision rules,
  ambiguity treatment, checks, and comparison procedures. Distinguish derived
  features from analytic judgments. Computation can constrain or test possible
  analyses. It cannot settle an interpretation.
- In `Results`, give the relevant denominator, outcomes, failures, negative
  results, disagreement, and unresolved cases. State the comparison or baseline
  that makes each result interpretable.
- In `Limitations`, state the inferential boundary of the outputs, the theory
  embedded in the textual or analytical unit, missing information, and cases
  where agreement measures source consistency rather than validation.
- In `Reproducibility`, identify versions, dependencies, source access,
  transformations, and commands required to rebuild results. Distinguish a
  repeatable computation from a transparent interpretive decision.
- In `Installation` and `Usage`, provide the shortest complete path to install,
  run, and check the repository.
- In `References`, give complete, verified bibliographic entries for works that
  inform the repository. Use `Citation` only for a separate preferred citation.

### Technical precision

- Write a concise technical research account that permits scholarly review from
  the README. Make the data model, operational choices, evidence, and
  inferential boundary intelligible without opening the code.
- Use precise named entities and quantities. Name files, fields, versions,
  algorithms, units, thresholds, metrics, and denominators when they carry the
  argument. Link primary resources that identify a source, corpus, or tool.
- Give the rationale for every non-obvious analytical choice. Relate the choice
  to the structure of the material, the research problem, or a verified source.
- Make each result traceable to a defined input, operation, and comparison.
  Separate empirical findings from interpretive inferences.
- Keep the explanatory sequence visible: material, representation, operation,
  result, and scope. Concision comes from information density. It does not omit
  technical conditions or qualifying evidence.

### Prose and research guidance

- Use compact factual prose. Exclude generic research framing, promotional
  claims, rhetorical questions, and template filler. No em dashes, semicolons,
  dash-interjections, or contrastive negation formulas.
- Consult `tehillim-literature` and its `van-peursen` skill before drafting a
  Tehillim README. Apply its approach through the treatment of data, formal
  analysis, interpretation, ambiguity, disagreement, and falsifiability. Do not
  name or cite van Peursen unless the README concerns his work.
