import { createFireworks } from '@ai-sdk/fireworks';
import { createOpenAI } from '@ai-sdk/openai';
import {
  extractReasoningMiddleware,
  generateObject,
  GenerateObjectResult,
  LanguageModelV1,
  wrapLanguageModel,
} from 'ai';
import { getEncoding } from 'js-tiktoken';
import { ProxyAgent, fetch as undiciFetch } from 'undici';

import { RecursiveCharacterTextSplitter } from './text-splitter';

// Proxy support — uses undici fetch + ProxyAgent so HTTPS through an HTTP proxy works
const proxyAgent = process.env.HTTPS_PROXY
  ? new ProxyAgent(process.env.HTTPS_PROXY)
  : undefined;
const customFetch = proxyAgent
  ? (url: RequestInfo | URL, init?: RequestInit) =>
      undiciFetch(url, { ...init, dispatcher: proxyAgent } as any)
  : undefined;

// Providers
const openai = process.env.OPENAI_KEY
  ? createOpenAI({
      apiKey: process.env.OPENAI_KEY,
      baseURL: process.env.OPENAI_ENDPOINT || 'https://api.openai.com/v1',
      fetch: customFetch,
    })
  : undefined;

const fireworks = process.env.FIREWORKS_KEY
  ? createFireworks({
      apiKey: process.env.FIREWORKS_KEY,
    })
  : undefined;

const customModel = process.env.CUSTOM_MODEL
  ? openai?.(process.env.CUSTOM_MODEL)
  : undefined;

// Models

const o3MiniModel = openai?.('o3-mini', {
  reasoningEffort: 'medium',
  structuredOutputs: true,
});

const deepSeekR1Model = fireworks
  ? wrapLanguageModel({
      model: fireworks(
        'accounts/fireworks/models/deepseek-r1',
      ) as LanguageModelV1,
      middleware: extractReasoningMiddleware({ tagName: 'think' }),
    })
  : undefined;

export function getModel(): LanguageModelV1 {
  if (customModel) {
    return customModel;
  }

  const model = deepSeekR1Model ?? o3MiniModel;
  if (!model) {
    throw new Error('No model found');
  }

  return model as LanguageModelV1;
}

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(
  prompt: string,
  contextSize = Number(process.env.CONTEXT_SIZE) || 128_000,
) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}

/**
 * Wrapper around generateObject that handles models with poor structured output
 * support (e.g. deepseek-v4-pro). Retries on failure and attempts manual
 * JSON.parse for double-serialized or malformed responses.
 */
export async function safeGenerateObject<T>(
  params: Parameters<typeof generateObject>[0],
  maxRetries = 2,
): Promise<GenerateObjectResult<T>> {
  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return (await generateObject(params)) as GenerateObjectResult<T>;
    } catch (e: any) {
      lastError = e;

      if (e?.name !== 'AI_NoObjectGeneratedError') {
        throw e;
      }

      // Try manual JSON parse recovery from e.text (double-serialization case)
      const recovered = tryRecoverFromText<T>(e, params);
      if (recovered !== null) {
        return recovered;
      }

      // If we have retries left, try again; otherwise throw
      if (attempt < maxRetries) {
        continue;
      }
    }
  }

  throw lastError;
}

/**
 * Attempt to recover a valid object from AI_NoObjectGeneratedError by manually
 * parsing the raw text. Handles:
 * - Double-serialized JSON (string containing JSON string)
 * - Model returning JSON as a plain string instead of object
 * Returns null if recovery is not possible.
 */
function tryRecoverFromText<T>(
  e: any,
  params: Parameters<typeof generateObject>[0],
): GenerateObjectResult<T> | null {
  const raw = e?.text;
  if (typeof raw !== 'string') {
    return null;
  }

  let parsed: any;
  try {
    parsed = JSON.parse(raw);
    // If parsed is still a string, parse one more level
    if (typeof parsed === 'string') {
      parsed = JSON.parse(parsed);
    }
  } catch {
    return null;
  }

  // Only accept if parsed is a non-null object
  if (typeof parsed !== 'object' || parsed === null) {
    return null;
  }

  // Try schema validation with safeParse — don't throw on mismatch
  const schema = (params as any).schema;
  if (schema?.safeParse) {
    const result = schema.safeParse(parsed);
    if (!result.success) {
      return null;
    }
    parsed = result.data;
  }

  return {
    object: parsed as T,
    finishReason: 'stop',
    usage: e?.usage ?? { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    response: e?.response ?? {},
  } as GenerateObjectResult<T>;
}
