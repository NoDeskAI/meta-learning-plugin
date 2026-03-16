import { writeFileSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { dump as yamlDump } from "js-yaml";

export type TriggerReason = "error_recovery" | "user_correction" | "new_tool" | "efficiency_anomaly";

export type SignalData = {
  signal_id: string;
  timestamp: string;
  session_id: string;
  memory_date: string;
  trigger_reason: TriggerReason;
  keywords: string[];
  task_summary: string;
  error_snapshot: string | null;
  resolution_snapshot: string | null;
  user_feedback: string | null;
  step_count: number;
  processed: boolean;
};

type MessageObj = {
  role?: string;
  content?: string | Array<{ type?: string; text?: string }>;
};

export function analyzeMessagesForSignal(
  messages: unknown[],
  sessionId: string | undefined,
): SignalData | null {
  const analysis = extractConversationSignals(messages);

  let trigger: TriggerReason | null = null;

  if (analysis.errorsEncountered.length > 0 && analysis.errorsFixed) {
    trigger = "error_recovery";
  } else if (analysis.userCorrections.length > 0) {
    trigger = "user_correction";
  } else if (analysis.toolCount > 10) {
    trigger = "efficiency_anomaly";
  }

  if (!trigger) return null;

  const signalId = generateSignalId();
  const now = new Date();

  let errorSnapshot: string | null = null;
  if (trigger === "error_recovery" && analysis.errorsEncountered.length > 0) {
    errorSnapshot = analysis.errorsEncountered[0].slice(0, 500);
  }

  let userFeedback: string | null = null;
  if (analysis.userCorrections.length > 0) {
    userFeedback = analysis.userCorrections[0].slice(0, 500);
  }

  return {
    signal_id: signalId,
    timestamp: now.toISOString(),
    session_id: sessionId ?? "unknown",
    memory_date: now.toISOString().split("T")[0],
    trigger_reason: trigger,
    keywords: extractKeywords(analysis),
    task_summary: analysis.taskSummary.slice(0, 200),
    error_snapshot: errorSnapshot,
    resolution_snapshot: null,
    user_feedback: userFeedback,
    step_count: analysis.toolCount,
    processed: false,
  };
}

type ConversationAnalysis = {
  taskSummary: string;
  errorsEncountered: string[];
  errorsFixed: boolean;
  userCorrections: string[];
  toolCount: number;
};

const CORRECTION_PATTERNS = [
  /不对/,
  /不是/,
  /错了/,
  /应该是/,
  /我说过/,
  /不要/,
  /wrong/i,
  /incorrect/i,
  /no,?\s+(please|you should|that's not|it should)/i,
  /actually,?\s/i,
];

const ERROR_PATTERNS = [
  /error[:\s]/i,
  /exception[:\s]/i,
  /failed[:\s]/i,
  /traceback/i,
  /\bTS\d{4}\b/,
  /\bTypeError\b/,
  /\bSyntaxError\b/,
  /\bReferenceError\b/,
  /command\s+not\s+found/i,
  /permission\s+denied/i,
  /ENOENT/,
  /EACCES/,
];

const FIX_INDICATORS = [
  /fix/i,
  /fixed/i,
  /resolved/i,
  /solved/i,
  /修复/,
  /解决/,
  /搞定/,
  /完成/,
];

function extractConversationSignals(messages: unknown[]): ConversationAnalysis {
  let taskSummary = "";
  const errorsEncountered: string[] = [];
  let errorsFixed = false;
  const userCorrections: string[] = [];
  let toolCount = 0;

  for (const raw of messages) {
    if (!raw || typeof raw !== "object") continue;
    const msg = raw as MessageObj;
    const text = extractTextContent(msg);

    if (msg.role === "user") {
      if (!taskSummary) {
        taskSummary = text.slice(0, 200);
      }
      if (CORRECTION_PATTERNS.some((p) => p.test(text))) {
        userCorrections.push(text.slice(0, 300));
      }
    }

    if (msg.role === "assistant" || msg.role === "tool") {
      for (const pat of ERROR_PATTERNS) {
        if (pat.test(text)) {
          const match = text.match(pat);
          if (match) {
            const start = Math.max(0, (match.index ?? 0) - 50);
            errorsEncountered.push(text.slice(start, start + 300));
          }
          break;
        }
      }

      if (FIX_INDICATORS.some((p) => p.test(text))) {
        errorsFixed = true;
      }
    }

    if (msg.role === "assistant") {
      const toolCallCount = countToolCalls(raw);
      toolCount += toolCallCount;
    }
  }

  return { taskSummary, errorsEncountered, errorsFixed, userCorrections, toolCount };
}

function extractTextContent(msg: MessageObj): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .filter((b): b is { type: string; text: string } => b?.type === "text" && typeof b.text === "string")
      .map((b) => b.text)
      .join(" ");
  }
  return "";
}

function countToolCalls(msg: unknown): number {
  if (!msg || typeof msg !== "object") return 0;
  const m = msg as Record<string, unknown>;
  if (Array.isArray(m.content)) {
    return (m.content as Array<Record<string, unknown>>).filter(
      (b) => b?.type === "tool_use" || b?.type === "tool_call",
    ).length;
  }
  return 0;
}

function extractKeywords(analysis: ConversationAnalysis): string[] {
  const keywords: string[] = [];

  for (const error of analysis.errorsEncountered.slice(0, 3)) {
    for (const token of error.split(/\s+/)) {
      const cleaned = token.replace(/^[.:,;()\[\]{}"']+|[.:,;()\[\]{}"']+$/g, "");
      if (
        cleaned.length > 2 &&
        (/^[A-Z]/.test(cleaned) || /\d/.test(cleaned)) &&
        !keywords.includes(cleaned)
      ) {
        keywords.push(cleaned);
        if (keywords.length >= 10) break;
      }
    }
    if (keywords.length >= 10) break;
  }

  if (keywords.length === 0) {
    for (const word of analysis.taskSummary.split(/\s+/)) {
      const cleaned = word.replace(/^[.:,;()\[\]{}"']+|[.:,;()\[\]{}"']+$/g, "").toLowerCase();
      if (cleaned.length > 3 && !keywords.includes(cleaned)) {
        keywords.push(cleaned);
        if (keywords.length >= 5) break;
      }
    }
  }

  return keywords.slice(0, 15);
}

function generateSignalId(): string {
  const now = new Date();
  const dateStr = now.toISOString().split("T")[0].replace(/-/g, "");
  const rand = String(Math.floor(Math.random() * 999) + 1).padStart(3, "0");
  return `sig-${dateStr}-${rand}`;
}

export function writeSignal(workspaceRoot: string, signalBufferDir: string, signal: SignalData): string {
  const bufDir = resolve(workspaceRoot, signalBufferDir);
  mkdirSync(bufDir, { recursive: true });

  const dedupedId = deduplicateSignalId(bufDir, signal.signal_id);
  signal.signal_id = dedupedId;

  const filePath = resolve(bufDir, `${dedupedId}.yaml`);
  writeFileSync(filePath, yamlDump(signal, { lineWidth: -1 }), "utf-8");
  return filePath;
}

function deduplicateSignalId(bufDir: string, signalId: string): string {
  if (!existsSync(bufDir)) return signalId;
  const existing = readdirSync(bufDir).filter((f) => f.startsWith(signalId.slice(0, 12)));
  if (existing.length === 0) return signalId;

  const dateStr = signalId.split("-")[1];
  const nextNum = String(existing.length + 1).padStart(3, "0");
  return `sig-${dateStr}-${nextNum}`;
}
