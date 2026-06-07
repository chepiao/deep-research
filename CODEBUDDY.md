# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Project Overview

Open Deep Research — an AI-powered research assistant that performs iterative, deep research on any topic by combining search engines (Firecrawl), web scraping, and LLMs. Goal is to stay under 500 LoC for simplicity.

## Commands

```bash
# Install dependencies
npm install

# Run CLI research assistant (loads .env.local automatically)
npm start

# Run API server (Express, default port 3051)
npm run api

# Format source files
npm run format

# Run tests (uses Node.js built-in test runner, currently only text-splitter tests)
npx tsx --test src/ai/text-splitter.test.ts

# Docker
docker compose up -d
docker exec -it deep-research npm run docker
```

There is no build step — `tsx` runs TypeScript directly. No linter is configured; formatting is handled by Prettier.

## Environment Variables

Required in `.env.local`:

- `FIRECRAWL_KEY` — Firecrawl API key for web search/scraping
- `OPENAI_KEY` — OpenAI API key (uses o3-mini by default)

Optional:

- `FIRECRAWL_BASE_URL` — Self-hosted Firecrawl URL
- `FIRECRAWL_CONCURRENCY` — Max concurrent Firecrawl requests (default: 2)
- `CONTEXT_SIZE` — Token context limit for trimPrompt (default: 128000)
- `OPENAI_ENDPOINT` — Custom OpenAI-compatible endpoint (e.g., OpenRouter, local LLM)
- `CUSTOM_MODEL` — Model string when using custom endpoint
- `FIREWORKS_KEY` — Fireworks API key; when set, uses DeepSeek R1 instead of o3-mini
- `PORT` — API server port (default: 3051)

## Architecture

### Core Flow (CLI: `src/run.ts`)

1. User enters query + breadth/depth parameters
2. For "report" mode: `feedback.ts` generates follow-up questions → user answers → combined into enriched query
3. `deep-research.ts` runs iterative research loop
4. Output written to `report.md` (long report) or `answer.md` (concise answer)

### API Server (`src/api.ts`)

Express server with two endpoints:
- `POST /api/research` — returns concise answer + learnings + URLs
- `POST /api/generate-report` — returns full markdown report

### Deep Research Engine (`src/deep-research.ts`)

The recursive research loop:
- `generateSerpQueries()` — LLM generates SERP queries from research goal + prior learnings
- `firecrawl.search()` — executes searches concurrently (bounded by `pLimit`)
- `processSerpResult()` — LLM extracts learnings + follow-up questions from search results
- `deepResearch()` — recursively calls itself with `depth - 1` and `breadth / 2` until depth reaches 0
- `writeFinalReport()` / `writeFinalAnswer()` — LLM generates final output from accumulated learnings

All LLM calls use Vercel AI SDK's `generateObject` with Zod schemas for structured output.

### AI Providers (`src/ai/providers.ts`)

Model selection priority: `CUSTOM_MODEL` > `FIREWORKS_KEY` (DeepSeek R1) > `OPENAI_KEY` (o3-mini).

`trimPrompt()` uses tiktoken (o200k_base encoding) to truncate prompts to context size, using `RecursiveCharacterTextSplitter` for clean splits.

### Supporting Files

- `src/prompt.ts` — system prompt for all LLM calls
- `src/feedback.ts` — generates follow-up clarifying questions
- `src/ai/text-splitter.ts` — recursive character text splitter (LangChain-style)

## Key Patterns

- All LLM interactions go through `getModel()` from `src/ai/providers.ts` and use Vercel AI SDK (`ai` package) with Zod schemas
- Firecrawl is used for both search and markdown extraction; configured as a singleton at module level
- `ConcurrencyLimit` controls parallel Firecrawl requests; free-tier users may need to set to 1
- Output files (`report.md`, `answer.md`) are in `.gitignore`
