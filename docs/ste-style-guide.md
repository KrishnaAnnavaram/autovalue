# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of autovalue. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **listing** | One car offered for sale, one row of the listings table | ad, post, car record |
| **order** | One delivery, one row of the deliveries table | shipment, trip, job |
| **listed price** | The price that the seller asks for | asking value, offer price |
| **price** | The sale price label of a listing | value, cost |
| **delivery time** | The `delivery_minutes` label of an order | ETA (except the task name `eta`), duration |
| **task** | `price` or `eta`: one table, one target, one feature builder | job, problem |
| **band** | The low, middle and high values of one estimate | range, interval (except "bootstrap interval") |
| **quantile model** | The gradient-boosting model of one quantile | GBM (except the report name `quantile_gbm`) |
| **baseline** | The group-median model or the ridge model | benchmark, reference model |
| **noise floor** | The error of the label without noise | oracle error, ceiling |
| **calibration** | The change of the band on the validation part | tuning, adjustment |
| **coverage** | The fraction of actual values inside the band | hit rate, recall (for bands) |
| **time split** | The cut of rows into train, validation and test parts by time | holdout, partition |
| **validation part** | The middle time period, for calibration and feature ranking | dev set, tuning set |
| **test part** | The last time period, for the final report only | holdout set, eval set |
| **approval rule** | The rule that gives `auto_approve`, `review_low` or `review_high` | policy, gate |
| **decision** | One output of the approval rule | verdict, outcome |
| **adapter** | A function that maps a public dataset to the project schema | converter, parser |
| **model file** | The `.joblib` file with a calibrated quantile model | checkpoint, artefact |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Check a table against its schema and drop the rows that are not valid |
| **coerce** | Change the type of each column of new rows, with no dropped row |
| **split** | Cut the rows into train, validation and test parts |
| **fit** | Learn the parameters of a pipeline from the training part |
| **calibrate** | Calculate the band offset on the validation part |
| **evaluate** | Calculate the metrics on the test part |
| **estimate** | Give the band of a new row |
| **approve** | Give the decision for a listed price |
| **simulate** | Run the seeded negotiation episodes |
