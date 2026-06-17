export function chunkByChar(
  text: string,
  chunkSize: number = 150,
  chunkOverlap: number = 20,
): string[] {
  const chunks: string[] = [];
  let startIdx = 0;

  while (startIdx < text.length) {
    const endIdx = Math.min(startIdx + chunkSize, text.length);

    const chunkText = text.slice(startIdx, endIdx);
    chunks.push(chunkText);

    startIdx = endIdx < text.length ? endIdx - chunkOverlap : text.length;
  }

  return chunks;
}

export function chunkBySentence(
  text: string,
  maxSentencesPerChunk: number = 5,
  overlapSentences: number = 1,
): string[] {
  const sentences = text.split(/(?<=[.!?])\s+/);

  const chunks: string[] = [];
  let startIdx = 0;

  while (startIdx < sentences.length) {
    const endIdx = Math.min(startIdx + maxSentencesPerChunk, sentences.length);

    const currentChunk = sentences.slice(startIdx, endIdx);
    chunks.push(currentChunk.join(" "));

    startIdx += maxSentencesPerChunk - overlapSentences;

    if (startIdx < 0) {
      startIdx = 0;
    }
  }

  return chunks;
}

export function chunkBySection(documentText: string): string[] {
  const pattern = /\n## /;
  return documentText.split(pattern);
}
