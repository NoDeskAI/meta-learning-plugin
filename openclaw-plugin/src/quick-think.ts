import type { TaxonomyEntry, ErrorTaxonomy } from "./taxonomy-loader.js";
import { buildKeywordIndex } from "./taxonomy-loader.js";

const IRREVERSIBLE_PATTERNS = [
  "rm -rf",
  "drop table",
  "force push",
  "git push --force",
  "delete from",
  "overwrite",
  "truncate",
  "format disk",
  "fdisk",
  "mkfs",
];

export type QuickThinkResult = {
  hit: boolean;
  matchedSignals: string[];
  matchedTaxonomyEntries: string[];
  riskLevel: "none" | "low" | "medium" | "high";
};

export function quickThinkEvaluate(
  taxonomy: ErrorTaxonomy | null,
  userMessage: string,
  toolsUsed: string[] = [],
): QuickThinkResult {
  const matchedSignals: string[] = [];
  const matchedTaxonomyIds: string[] = [];

  const textLower = userMessage.toLowerCase();
  const searchCorpus = [textLower, ...toolsUsed.map((t) => t.toLowerCase())].join(" ");

  if (taxonomy) {
    const kwIndex = buildKeywordIndex(taxonomy);
    const seenIds = new Set<string>();
    for (const [keyword, entries] of kwIndex) {
      if (searchCorpus.includes(keyword)) {
        for (const entry of entries) {
          if (!seenIds.has(entry.id)) {
            matchedTaxonomyIds.push(entry.id);
            seenIds.add(entry.id);
          }
        }
      }
    }
    if (matchedTaxonomyIds.length > 0) {
      matchedSignals.push("keyword_taxonomy_hit");
    }
  }

  if (IRREVERSIBLE_PATTERNS.some((p) => searchCorpus.includes(p))) {
    matchedSignals.push("irreversible_operation");
  }

  const hit = matchedSignals.length > 0;
  const riskLevel = assessRiskLevel(matchedSignals);

  return { hit, matchedSignals, matchedTaxonomyEntries: matchedTaxonomyIds, riskLevel };
}

function assessRiskLevel(signals: string[]): QuickThinkResult["riskLevel"] {
  if (signals.includes("irreversible_operation")) return "high";
  if (signals.length >= 2) return "medium";
  if (signals.length > 0) return "low";
  return "none";
}

export function formatRiskWarning(result: QuickThinkResult, taxonomy: ErrorTaxonomy | null): string {
  const lines: string[] = [];
  lines.push(`<meta-learning-risk-assessment level="${result.riskLevel}">`);

  if (result.matchedSignals.includes("irreversible_operation")) {
    lines.push("WARNING: This task involves irreversible operations. Generate a rollback plan before executing.");
  }

  if (result.matchedSignals.includes("keyword_taxonomy_hit") && taxonomy) {
    lines.push("Known error patterns detected from past experience:");
    for (const entryId of result.matchedTaxonomyEntries.slice(0, 3)) {
      const entry = findEntryById(taxonomy, entryId);
      if (entry) {
        lines.push(`  - [${entry.id}] ${entry.name}: ${entry.prevention}`);
      }
    }
  }

  lines.push("</meta-learning-risk-assessment>");
  return lines.join("\n");
}

function findEntryById(taxonomy: ErrorTaxonomy, id: string): TaxonomyEntry | null {
  for (const domain of Object.values(taxonomy.taxonomy)) {
    for (const entries of Object.values(domain)) {
      for (const entry of entries) {
        if (entry.id === id) return entry;
      }
    }
  }
  return null;
}
