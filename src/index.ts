import { readFileSync } from "node:fs";
import "dotenv/config";
import { chunkBySection } from "./lib/chunking";
import { generateEmbedding } from "./lib/embeddings";
import { VectorIndex } from "./lib/vectorIndex";

const text = readFileSync("./report.md", "utf-8");

const chunks = chunkBySection(text);

const store = new VectorIndex("cosine", (texts: string[]) =>
  generateEmbedding(texts, "voyage-3-large", "document"),
);
await store.addDocuments(chunks.map((chunk) => ({ content: chunk })));

const userQuestion = "What did the software engineering dept do this year?";
const queryVector = await generateEmbedding(
  userQuestion,
  "voyage-3-large",
  "query",
);

const results = await store.search(queryVector, 2);

results.forEach(([doc, distance], i) => {
  console.log(
    `Match ${i + 1} (distance: ${distance.toFixed(4)}):\n${doc.content}\n----\n`,
  );
});
