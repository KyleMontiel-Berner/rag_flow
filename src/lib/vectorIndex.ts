export interface VectorDocument {
  content: string;
  [key: string]: unknown;
}

export type EmbeddingFn = (texts: string[]) => Promise<number[][]>;

export class VectorIndex {
  private vectors: number[][] = [];
  private documents: VectorDocument[] = [];
  private vectorDim: number | null = null;
  private distanceMetric: "cosine" | "euclidean";
  private embeddingFn?: EmbeddingFn;

  constructor(
    distanceMetric: "cosine" | "euclidean" = "cosine",
    embeddingFn?: EmbeddingFn,
  ) {
    if (distanceMetric !== "cosine" && distanceMetric !== "euclidean") {
      throw new Error("distanceMetric must be 'cosine' or 'euclidean'");
    }
    this.distanceMetric = distanceMetric;
    this.embeddingFn = embeddingFn;
  }

  async addDocument(document: VectorDocument): Promise<void> {
    if (!this.embeddingFn)
      throw new Error("Embedding function not provided during initialization.");
    if (typeof document.content !== "string")
      throw new TypeError("Document 'content' must be a string.");

    const [vector] = await this.embeddingFn([document.content]);
    this.addVector(vector, document);
  }

  async addDocuments(documents: VectorDocument[]): Promise<void> {
    if (!this.embeddingFn)
      throw new Error("Embedding function not provided during initialization.");

    const contents = documents.map((doc) => {
      if (typeof doc.content !== "string")
        throw new TypeError("Document 'content' must be a string.");
      return doc.content;
    });

    const vectors = await this.embeddingFn(contents);
    documents.forEach((doc, i) => this.addVector(vectors[i], doc));
  }

  async search(
    query: string | number[],
    k: number = 1,
  ): Promise<Array<[VectorDocument, number]>> {
    if (this.vectors.length === 0) return [];

    let queryVector: number[];
    if (typeof query === "string") {
      if (!this.embeddingFn)
        throw new Error("Embedding function not provided for string query.");
      [queryVector] = await this.embeddingFn([query]);
    } else if (
      Array.isArray(query) &&
      query.every((x) => typeof x === "number")
    ) {
      queryVector = query;
    } else {
      throw new TypeError(
        "Query must be either a string or an array of numbers.",
      );
    }

    if (this.vectorDim === null) return [];
    if (queryVector.length !== this.vectorDim) {
      throw new Error(
        `Query vector dimension mismatch. Expected ${this.vectorDim}, got ${queryVector.length}`,
      );
    }
    if (k <= 0) throw new Error("k must be a positive integer.");

    const distFn =
      this.distanceMetric === "cosine"
        ? this.cosineDistance.bind(this)
        : this.euclideanDistance.bind(this);

    const distances: Array<[number, VectorDocument]> = this.vectors.map(
      (v, i) => [distFn(queryVector, v), this.documents[i]],
    );
    distances.sort((a, b) => a[0] - b[0]);

    return distances.slice(0, k).map(([dist, doc]) => [doc, dist]);
  }

  addVector(vector: number[], document: VectorDocument): void {
    if (this.vectors.length === 0) {
      this.vectorDim = vector.length;
    } else if (vector.length !== this.vectorDim) {
      throw new Error(
        `Inconsistent vector dimension. Expected ${this.vectorDim}, got ${vector.length}`,
      );
    }
    this.vectors.push([...vector]);
    this.documents.push(document);
  }

  private euclideanDistance(vec1: number[], vec2: number[]): number {
    return Math.sqrt(vec1.reduce((sum, v, i) => sum + (v - vec2[i]) ** 2, 0));
  }
  private dotProduct(vec1: number[], vec2: number[]): number {
    return vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
  }
  private magnitude(vec: number[]): number {
    return Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0));
  }
  private cosineDistance(vec1: number[], vec2: number[]): number {
    const mag1 = this.magnitude(vec1);
    const mag2 = this.magnitude(vec2);
    if (mag1 === 0 && mag2 === 0) return 0.0;
    if (mag1 === 0 || mag2 === 0) return 1.0;
    const cosineSimilarity = Math.max(
      -1,
      Math.min(1, this.dotProduct(vec1, vec2) / (mag1 * mag2)),
    );
    return 1.0 - cosineSimilarity;
  }

  get length(): number {
    return this.vectors.length;
  }
  toString(): string {
    return `VectorIndex(count=${this.length}, dim=${this.vectorDim}, metric='${this.distanceMetric}')`;
  }
}
