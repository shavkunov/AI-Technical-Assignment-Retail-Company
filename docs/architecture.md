# Architecture

Retail data analysis agent on Google Cloud for non-technical executives. How it works in detail, requirement by
requirement, and the assumptions behind it: [technical-design.md](technical-design.md). The same views as
standalone pages to open in a browser: [architecture.html](architecture.html), [extensibility.html](extensibility.html).

Five views, each answering one question. Colours: cyan = client, rose = guardrail / security, green = service,
amber = managed service, violet = data store, orange = queue. Dashed arrows are asynchronous.

## 1. What runs where

![Containers: what runs where](diagrams/01-containers.svg)

Request path: ① ask ② HTTPS ③ verified user ④ only data questions pass ⑤ SQL checked ⑥ query runs ⑦ rows
⑧ PII removed ⑨ answer. C: a delete waits for the user's typed yes.

## 2. What the guardrails stop

![Guardrails: what each layer stops](diagrams/02-guardrails.svg)

## 3. How the agent is assembled

![How the agent is assembled](diagrams/03-config.svg)

## 4. One question, start to finish

![User flow](diagrams/04-flow.svg)

## 5. Where the agent can grow

The assignment asks for a system "easily extendable for new capabilities (generating graphs, sending reports via
mail, searching the web for trends, etc)". Solid = built or designed, dashed = extension point; the three tagged
"assignment" are the ones the task names. Details: [technical-design.md](technical-design.md#7-extensibility).

![Extension points](diagrams/05-extensibility.svg)
