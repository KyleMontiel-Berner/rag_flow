import { VoyageAIClient } from "voyageai";

const client = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

export async function generateEmbedding(
  chunks: string,
  model?: string,
  inputType?: "query" | "document",
): Promise<number[]>;
export async function generateEmbedding(
  chunks: string[],
  model?: string,
  inputType?: "query" | "document",
): Promise<number[][]>;
export async function generateEmbedding(
  chunks: string | string[],
  model: string = "voyage-3-large",
  inputType: "query" | "document" = "query",
): Promise<number[] | number[][]> {
  const isList = Array.isArray(chunks);
  const input = isList ? chunks : [chunks];

  const result = await client.embed({ input, model, inputType });
  const embeddings = (result.data ?? []).map((item) => item.embedding ?? []);

  return isList ? embeddings : embeddings[0];
}
