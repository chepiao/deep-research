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
 * Wrapper around generateObject that handles models returning double-serialized
 * JSON strings (e.g. deepseek-v4-pro). Falls back to manual JSON.parse when the
 * SDK's schema validation fails due to the response being a string instead of object.
 */
export async function safeGenerateObject<T>(
  params: Parameters<typeof generateObject>[0],
): Promise<GenerateObjectResult<T>> {
  try {
    return await generateObject(params) as GenerateObjectResult<T>;
  } catch (e: any) {
    // Only handle the specific double-serialization issue
    if (e?.name !== 'AI_NoObjectGeneratedError') {
      throw e;
    }
    const raw = e?.text;
    if (typeof raw !== 'string') {
      throw e;
    }
    // Try to unwrap: the text might be a JSON string containing another JSON string
    let parsed: any;
    try {
      parsed = JSON.parse(raw);
      // If parsed is still a string, parse one more level
      if (typeof parsed === 'string') {
        parsed = JSON.parse(parsed);
      }
    } catch {
      throw e;
    }
    // Validate against schema if available
    const schema = (params as any).schema;
    if (schema?.parse) {
      parsed = schema.parse(parsed);
    }
    return {
      object: parsed as T,
      finishReason: 'stop',
      usage: e?.usage ?? { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      response: e?.response ?? {},
    } as GenerateObjectResult<T>;
  }
}
