import fs from "fs/promises";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import OpenAI from "openai";
import { Client, GatewayIntentBits } from "discord.js";

const execFileAsync = promisify(execFile);

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent
  ]
});

const MAX_REPLY = 1900;
const WORKSPACE_ROOT = "/workspace";
const AGENT_DIR = path.join(WORKSPACE_ROOT, ".agent");
const TASKS_FILE = path.join(AGENT_DIR, "tasks.json");
const USAGE_FILE = path.join(AGENT_DIR, "usage.json");
const AGENT_CHANNEL_NAME = process.env.AGENT_CHANNEL_NAME || "ai-agent";

const runtimeRouteOverrides = {
  chat: null,
  plan: null,
  execute: null
};

const HIGH_RISK_TOOL_NAMES = new Set([
  "create_vite_react_app",
  "npm_install",
  "npm_build",
  "npm_test",
  "run_node"
]);

const SAFE_TOOL_NAMES = new Set([
  "list_files",
  "read_file",
  "write_file",
  "edit_file",
  "web_search",
  "fetch_url"
]);

const HIGH_RISK_KEYWORDS = [
  "npm install",
  "npm run",
  "build",
  "ビルド",
  "test",
  "テスト",
  "run",
  "実行",
  "起動",
  "create vite",
  "vite",
  "react app",
  "アプリを作って",
  "新規アプリ",
  "server",
  "サーバー",
  "db",
  "database",
  "migration",
  "docker",
  "git push",
  "削除",
  "remove",
  "rm ",
  "sudo",
  "curl ",
  "wget "
];

const DEFAULT_PROVIDER = (process.env.LLM_DEFAULT_PROVIDER || "gemini").toLowerCase();

function envFirst(...names) {
  for (const name of names) {
    if (process.env[name]) return process.env[name];
  }
  return undefined;
}

function resolveTaskRoute(taskType) {
  if (taskType === "chat") {
    return {
      provider: (envFirst("MODEL_CHAT_PROVIDER") || DEFAULT_PROVIDER).toLowerCase(),
      model: envFirst("MODEL_CHAT", "OPENAI_MODEL_CHAT") || "gemini-3-flash-preview"
    };
  }

  if (taskType === "plan" || taskType === "dev_consult") {
    return {
      provider: (envFirst("MODEL_PLAN_PROVIDER") || DEFAULT_PROVIDER).toLowerCase(),
      model: envFirst("MODEL_PLAN", "OPENAI_MODEL_PLAN") || "gemini-3-flash-preview"
    };
  }

  if (taskType === "execute") {
    return {
      provider: (envFirst("MODEL_EXECUTE_PROVIDER") || DEFAULT_PROVIDER).toLowerCase(),
      model: envFirst("MODEL_EXECUTE", "OPENAI_MODEL_EXECUTE") || "gemini-3-flash-preview"
    };
  }

  return {
    provider: DEFAULT_PROVIDER,
    model: envFirst("MODEL_CHAT", "OPENAI_MODEL_CHAT") || "gemini-3-flash-preview"
  };
}

function normalizeProviderName(provider) {
  const p = String(provider || "").toLowerCase().trim();
  if (p === "claude") return "anthropic";
  return p;
}

function getEffectiveRoute(taskType) {
  const base = resolveTaskRoute(taskType);
  const override = runtimeRouteOverrides[taskType];
  if (!override) return base;
  return {
    provider: override.provider || base.provider,
    model: override.model || base.model
  };
}

function formatRouteLine(taskType) {
  const base = resolveTaskRoute(taskType);
  const effective = getEffectiveRoute(taskType);
  const overridden = Boolean(runtimeRouteOverrides[taskType]);

  if (!overridden) {
    return `${taskType}: ${effective.provider} / ${effective.model}`;
  }

  return `${taskType}: ${effective.provider} / ${effective.model} (runtime override, base=${base.provider}/${base.model})`;
}

function standardCommandGuide() {
  return [
    "利用可能なコマンド:",
    "",
    "通常利用:",
    `- #${AGENT_CHANNEL_NAME} では通常メッセージだけで利用できます`,
    "- 他チャンネルでは Bot へのメンション時のみ反応します",
    "",
    "基本コマンド:",
    "- help / ヘルプ",
    "  使い方を表示します",
    "- !status",
    "  直近のタスク一覧を表示します",
    "- !status task-xxxx",
    "  指定タスクの詳細を表示します",
    "- !approve task-xxxx",
    "  承認待ちタスクを実行します",
    "- !usage",
    "  usage 情報を表示します",
    "",
    "provider 切替:",
    "- !provider",
    "  現在の chat / plan / execute の割当を表示します",
    "- !provider chat <provider> <model>",
    "- !provider plan <provider> <model>",
    "- !provider execute <provider> <model>",
    "  実行中 bot の provider / model を切り替えます",
    "- !provider reset chat",
    "- !provider reset plan",
    "- !provider reset execute",
    "- !provider reset all",
    "  runtime override を解除します",
    "",
    "provider 例:",
    "- openai",
    "- gemini",
    "- anthropic",
    "- claude （内部では anthropic として扱います）",
    "- qwen",
    "",
    "注意:",
    "- qwen を使う場合は .env.openclaw に QWEN_API_KEY と QWEN_BASE_URL を設定してください"
  ].join("\n");
}

function standardProviderStatusText() {
  return [
    "現在の provider 設定:",
    formatRouteLine("chat"),
    formatRouteLine("plan"),
    formatRouteLine("execute")
  ].join("\n");
}

function getProviderConfig(provider) {
  switch (provider) {
    case "openai":
      if (!process.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY が未設定です");
      }
      return {
        apiKey: process.env.OPENAI_API_KEY,
        baseURL: undefined
      };

    case "gemini":
      if (!process.env.GEMINI_API_KEY) {
        throw new Error("GEMINI_API_KEY が未設定です");
      }
      return {
        apiKey: process.env.GEMINI_API_KEY,
        baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
      };

    case "anthropic":
    case "claude":
      if (!process.env.ANTHROPIC_API_KEY) {
        throw new Error("ANTHROPIC_API_KEY が未設定です");
      }
      return {
        apiKey: process.env.ANTHROPIC_API_KEY,
        baseURL: "https://api.anthropic.com/v1/"
      };

    case "qwen":
      if (!process.env.QWEN_API_KEY) {
        throw new Error("QWEN_API_KEY が未設定です");
      }
      if (!process.env.QWEN_BASE_URL) {
        throw new Error("QWEN_BASE_URL が未設定です");
      }
      return {
        apiKey: process.env.QWEN_API_KEY,
        baseURL: process.env.QWEN_BASE_URL
      };

    default:
      throw new Error(`未対応 provider: ${provider}`);
  }
}

function makeClient(provider) {
  const conf = getProviderConfig(provider);
  return new OpenAI({
    apiKey: conf.apiKey,
    baseURL: conf.baseURL
  });
}

function clip(text, max = MAX_REPLY) {
  if (!text) return "";
  return text.length > max ? text.slice(0, max) + "\n...省略..." : text;
}

function safeJoin(relativePath = ".") {
  const resolved = path.resolve(WORKSPACE_ROOT, relativePath);
  const base = path.resolve(WORKSPACE_ROOT);
  if (!resolved.startsWith(base)) {
    throw new Error("workspace外にはアクセスできません");
  }
  return resolved;
}

function removeBotMention(text) {
  const mentionRegex = /<@!?\d+>/g;
  return text.replace(mentionRegex, "").trim();
}

function isAgentChannel(message) {
  return message.channel?.name === AGENT_CHANNEL_NAME;
}

function isMentioned(message) {
  return Boolean(client.user?.id) && message.mentions?.users?.has(client.user.id);
}

function shouldHandleMessage(message) {
  return isAgentChannel(message) || isMentioned(message);
}

function ruleBasedTaskSignal(text) {
  const t = text.toLowerCase();

  const directTaskKeywords = [
    "作って", "作成", "生成", "実装", "修正", "追加", "編集", "保存", "書いて",
    "build", "ビルド", "test", "テスト", "実行", "run", "npm install", "npm run",
    "apiを使", "apiつない", "api連携", "db", "database", "サーバー立て", "react",
    "next.js", "vite", "workspace", "ファイル", "コード", "プロジェクト", "アプリ"
  ];

  return directTaskKeywords.some(k => t.includes(k));
}

function estimateTaskSize(text) {
  const t = text.toLowerCase();

  const largeSignals = [
    "アプリ", "プロジェクト", "react", "next.js", "vite", "apiサーバー",
    "db", "database", "build", "テスト", "test", "複数ファイル", "npm install"
  ];

  const smallSignals = [
    "readme", "README", "単一ファイル", "1ファイル", "少し修正", "軽く修正",
    "テキスト作成", "メモ作成"
  ];

  if (largeSignals.some(k => t.includes(k))) return "large";
  if (smallSignals.some(k => t.includes(k))) return "small";

  if (text.length > 80) return "large";
  return "small";
}

function requiresApprovalByPolicy(text) {
  const t = String(text || "").toLowerCase();
  return HIGH_RISK_KEYWORDS.some(k => t.includes(k));
}

function getToolsForTask(task) {
  const approved = task?.approval_granted === true;
  if (approved) return TOOLS;
  return TOOLS.filter(t => SAFE_TOOL_NAMES.has(t.function.name));
}

async function ensureFiles() {
  await fs.mkdir(AGENT_DIR, { recursive: true });

  try {
    await fs.access(TASKS_FILE);
  } catch {
    await fs.writeFile(TASKS_FILE, "[]", "utf8");
  }

  try {
    await fs.access(USAGE_FILE);
  } catch {
    await fs.writeFile(
      USAGE_FILE,
      JSON.stringify(
        {
          started_at: new Date().toISOString(),
          requests: 0,
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          by_model: {}
        },
        null,
        2
      ),
      "utf8"
    );
  }
}

async function readJson(file, fallback) {
  try {
    const raw = await fs.readFile(file, "utf8");
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

async function writeJson(file, data) {
  await fs.writeFile(file, JSON.stringify(data, null, 2), "utf8");
}

async function readTasks() {
  return await readJson(TASKS_FILE, []);
}

async function writeTasks(tasks) {
  await writeJson(TASKS_FILE, tasks);
}

async function createTask(goal, author, mode = "auto") {
  const tasks = await readTasks();
  const id = "task-" + Date.now().toString(36);
  const task = {
    id,
    goal,
    author,
    mode,
    status: "planned",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    plan: [],
    notes: "",
    result: "",
    approval_granted: false
  };
  tasks.unshift(task);
  await writeTasks(tasks);
  return task;
}

async function updateTask(id, patch) {
  const tasks = await readTasks();
  const idx = tasks.findIndex(t => t.id === id);
  if (idx === -1) throw new Error(`task not found: ${id}`);
  tasks[idx] = {
    ...tasks[idx],
    ...patch,
    updated_at: new Date().toISOString()
  };
  await writeTasks(tasks);
  return tasks[idx];
}

async function getTask(id) {
  const tasks = await readTasks();
  return tasks.find(t => t.id === id) || null;
}

async function appendUsageFromChatResponse(response, provider, modelName) {
  const usageFile = await readJson(USAGE_FILE, {
    started_at: new Date().toISOString(),
    requests: 0,
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    by_model: {}
  });

  const inTok = response?.usage?.prompt_tokens ?? 0;
  const outTok = response?.usage?.completion_tokens ?? 0;
  const totalTok = response?.usage?.total_tokens ?? (Number(inTok) + Number(outTok));
  const key = `${provider}:${modelName || "unknown"}`;

  usageFile.requests += 1;
  usageFile.input_tokens += inTok;
  usageFile.output_tokens += outTok;
  usageFile.total_tokens += totalTok;

  if (!usageFile.by_model[key]) {
    usageFile.by_model[key] = {
      requests: 0,
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0
    };
  }

  usageFile.by_model[key].requests += 1;
  usageFile.by_model[key].input_tokens += inTok;
  usageFile.by_model[key].output_tokens += outTok;
  usageFile.by_model[key].total_tokens += totalTok;

  await writeJson(USAGE_FILE, usageFile);
}

function getMessageText(message) {
  if (!message) return "";

  const content = message.content;

  if (typeof content === "string") {
    return content.trim();
  }

  if (Array.isArray(content)) {
    const joined = content
      .map(part => {
        if (typeof part === "string") return part;
        if (part?.type === "text") return part.text || "";
        if (part?.type === "output_text") return part.text || "";
        if (typeof part?.content === "string") return part.content;
        return "";
      })
      .join("\n")
      .trim();

    if (joined) return joined;
  }

  if (typeof message?.text === "string" && message.text.trim()) {
    return message.text.trim();
  }

  if (typeof message?.output_text === "string" && message.output_text.trim()) {
    return message.output_text.trim();
  }

  if (typeof message?.refusal === "string" && message.refusal.trim()) {
    return message.refusal.trim();
  }

  return "";
}

function extractJsonObject(text) {
  const s = String(text || "").trim();
  if (!s) return null;

  try {
    return JSON.parse(s);
  } catch {}

  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    const candidate = s.slice(start, end + 1);
    try {
      return JSON.parse(candidate);
    } catch {}
  }

  return null;
}
function messagesToResponsesInput(messages = []) {
  return messages.map(msg => {
    if (msg.role === "system") {
      return {
        role: "system",
        content: [{ type: "input_text", text: String(msg.content || "") }]
      };
    }

    if (msg.role === "user") {
      return {
        role: "user",
        content: [{ type: "input_text", text: String(msg.content || "") }]
      };
    }

    if (msg.role === "assistant") {
      return {
        role: "assistant",
        content: [{ type: "output_text", text: String(msg.content || "") }]
      };
    }

    if (msg.role === "tool") {
      return {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `Tool result:\n${String(msg.content || "")}`
          }
        ]
      };
    }

    return {
      role: "user",
      content: [{ type: "input_text", text: String(msg.content || "") }]
    };
  });
}

function getResponsesOutputText(response) {
  if (!response) return "";

  if (typeof response.output_text === "string" && response.output_text.trim()) {
    return response.output_text.trim();
  }

  const parts = [];
  for (const item of response.output || []) {
    for (const c of item.content || []) {
      if (c.type === "output_text" && c.text) {
        parts.push(c.text);
      }
    }
  }

  return parts.join("\n").trim();
}

async function callLLM(taskType, { messages, tools, temperature = 0.2, max_tokens = 1200 } = {}) {
  const route = getEffectiveRoute(taskType);
  const sdk = makeClient(route.provider);

  const payload = {
    model: route.model,
    messages,
    temperature,
    max_completion_tokens: max_tokens
  };

  const joined = (messages || [])
    .map(m => String(m?.content || ""))
    .join("\n")
    .toLowerCase();

  const needsSearch =
    route.provider === "qwen" &&
    process.env.QWEN_ENABLE_WEB_SEARCH === "true" &&
    (taskType === "chat" || taskType === "plan") &&
    [
      "最新", "news", "料金", "価格", "release", "リリース",
      "api", "version", "モデル", "公開", "today", "2026", "現在"
    ].some(k => joined.includes(k));

  if (needsSearch) {
    payload.enable_search = true;
    payload.search_options = {
      search_strategy: "agent"
    };
  }

  if (tools?.length) {
    payload.tools = tools;
    payload.tool_choice = "auto";
  }

  const response = await sdk.chat.completions.create(payload);
  await appendUsageFromChatResponse(response, route.provider, route.model);

  console.log("callLLM payload:", JSON.stringify(payload, null, 2));
  console.log("callLLM response:", JSON.stringify(response, null, 2));

  return {
    route,
    response
  };
}

async function listFiles(relativePath = ".") {
  const target = safeJoin(relativePath);
  const entries = await fs.readdir(target, { withFileTypes: true });
  return entries.map(e => ({
    name: e.name,
    type: e.isDirectory() ? "dir" : "file"
  }));
}

async function readFile(relativePath) {
  const target = safeJoin(relativePath);
  const stat = await fs.stat(target);
  if (stat.size > 200_000) {
    throw new Error("ファイルが大きすぎます");
  }
  return await fs.readFile(target, "utf8");
}

function normalizeGitHubUrl(inputUrl) {
  const url = String(inputUrl || "").trim();

  if (!/^https?:\/\//i.test(url)) {
    throw new Error("http または https のURLを指定してください");
  }

  const u = new URL(url);

  if (
    u.hostname === "github.com" &&
    u.pathname.includes("/blob/")
  ) {
    const parts = u.pathname.split("/").filter(Boolean);
    // owner/repo/blob/branch/path/to/file
    if (parts.length >= 5 && parts[2] === "blob") {
      const owner = parts[0];
      const repo = parts[1];
      const branch = parts[3];
      const filePath = parts.slice(4).join("/");
      return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${filePath}`;
    }
  }

  return url;
}

function stripHtml(html) {
  return String(html || "")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();
}

async function fetchUrl(url) {
  const normalizedUrl = normalizeGitHubUrl(url);

  const res = await fetch(normalizedUrl, {
    headers: {
      "User-Agent": "discord-agent",
      "Accept": "text/plain, text/html;q=0.9, */*;q=0.8"
    }
  });

  if (!res.ok) {
    throw new Error(`fetch failed: ${res.status} ${res.statusText}`);
  }

  const contentType = res.headers.get("content-type") || "";
  const body = await res.text();

  let text = body;
  if (contentType.includes("text/html")) {
    text = stripHtml(body);
  }

  if (!text.trim()) {
    throw new Error("URLの本文を取得できませんでした");
  }

  return {
    url: normalizedUrl,
    content_type: contentType,
    content: text.slice(0, 200000)
  };
}

async function writeFile(relativePath, content) {
  const target = safeJoin(relativePath);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, content, "utf8");
  return `wrote ${relativePath}`;
}

async function editFile(relativePath, findText, replaceText) {
  const target = safeJoin(relativePath);
  const original = await fs.readFile(target, "utf8");
  if (!original.includes(findText)) {
    throw new Error("find_text が見つかりません");
  }
  const updated = original.replace(findText, replaceText);
  await fs.writeFile(target, updated, "utf8");
  return `edited ${relativePath}`;
}
async function webSearch(query) {
  if (process.env.WEB_SEARCH_ENABLED === "false") {
    throw new Error("WEB_SEARCH_ENABLED=false のため web_search は無効です");
  }

  if (!process.env.TAVILY_API_KEY) {
    throw new Error("TAVILY_API_KEY が未設定です");
  }

  const q = String(query || "").trim();
  if (!q) {
    throw new Error("query を指定してください");
  }

  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.TAVILY_API_KEY}`
    },
    body: JSON.stringify({
      query: q,
      topic: "general",
      search_depth: "basic",
      max_results: 5,
      include_answer: true,
      include_raw_content: false
    })
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`web_search failed: ${res.status} ${text}`);
  }

  const data = await res.json();

  return {
    query: q,
    answer: data.answer || "",
    results: (data.results || []).map(item => ({
      title: item.title,
      url: item.url,
      content: item.content
    }))
  };
}

async function createViteReactApp(app_name) {
  if (!/^[a-zA-Z0-9._-]+$/.test(app_name)) {
    throw new Error("app_name は英数字・._- のみ");
  }

  const target = safeJoin(app_name);

  try {
    await fs.access(target);
    throw new Error("同名ディレクトリが既に存在します");
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
  }

  const { stdout, stderr } = await execFileAsync(
    "npm",
    ["create", "vite@latest", app_name, "--", "--template", "react"],
    {
      cwd: WORKSPACE_ROOT,
      timeout: 120000,
      maxBuffer: 1024 * 1024 * 8
    }
  );

  return { stdout, stderr };
}

async function npmInstall(project_dir) {
  const cwd = safeJoin(project_dir);
  const { stdout, stderr } = await execFileAsync("npm", ["install"], {
    cwd,
    timeout: 120000,
    maxBuffer: 1024 * 1024 * 8
  });
  return { stdout, stderr };
}

async function npmBuild(project_dir) {
  const cwd = safeJoin(project_dir);
  const { stdout, stderr } = await execFileAsync("npm", ["run", "build"], {
    cwd,
    timeout: 120000,
    maxBuffer: 1024 * 1024 * 8
  });
  return { stdout, stderr };
}

async function npmTest(project_dir) {
  const cwd = safeJoin(project_dir);
  const { stdout, stderr } = await execFileAsync("npm", ["test"], {
    cwd,
    timeout: 120000,
    maxBuffer: 1024 * 1024 * 8
  });
  return { stdout, stderr };
}

async function runNode(project_dir, entry = "index.js") {
  const cwd = safeJoin(project_dir);
  const { stdout, stderr } = await execFileAsync("node", [entry], {
    cwd,
    timeout: 30000,
    maxBuffer: 1024 * 1024 * 4
  });
  return { stdout, stderr };
}

const TOOLS = [
  {
    type: "function",
    function: {
      name: "list_files",
      description: "workspace配下のファイル一覧を取得する",
      parameters: {
        type: "object",
        properties: {
          relative_path: { type: "string" }
        },
        additionalProperties: false
      }
    }
  },
  {
  type: "function",
  function: {
    name: "fetch_url",
    description: "公開URLの本文を取得する。GitHubのblob URLも raw URL に変換して読める",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string" }
      },
      required: ["url"],
      additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "workspace配下のテキストファイルを読む",
      parameters: {
        type: "object",
        properties: {
          relative_path: { type: "string" }
        },
        required: ["relative_path"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description: "workspace配下にテキストファイルを書き込む",
      parameters: {
        type: "object",
        properties: {
          relative_path: { type: "string" },
          content: { type: "string" }
        },
        required: ["relative_path", "content"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "edit_file",
      description: "ファイル内の一部テキストを置換する",
      parameters: {
        type: "object",
        properties: {
          relative_path: { type: "string" },
          find_text: { type: "string" },
          replace_text: { type: "string" }
        },
        required: ["relative_path", "find_text", "replace_text"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "create_vite_react_app",
      description: "Vite React アプリを新規作成する",
      parameters: {
        type: "object",
        properties: {
          app_name: { type: "string" }
        },
        required: ["app_name"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "npm_install",
      description: "指定ディレクトリで npm install を実行する",
      parameters: {
        type: "object",
        properties: {
          project_dir: { type: "string" }
        },
        required: ["project_dir"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "npm_build",
      description: "指定ディレクトリで npm run build を実行する",
      parameters: {
        type: "object",
        properties: {
          project_dir: { type: "string" }
        },
        required: ["project_dir"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "npm_test",
      description: "指定ディレクトリで npm test を実行する",
      parameters: {
        type: "object",
        properties: {
          project_dir: { type: "string" }
        },
        required: ["project_dir"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "run_node",
      description: "指定ディレクトリで node を実行する",
      parameters: {
        type: "object",
        properties: {
          project_dir: { type: "string" },
          entry: { type: "string" }
        },
        required: ["project_dir"],
        additionalProperties: false
      }
    }
  },
  {
  type: "function",
  function: {
    name: "web_search",
    description: "最新情報や外部情報を調べるためにWeb検索を実行する",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string" }
      },
      required: ["query"],
      additionalProperties: false
      }
    }
  }
];

async function handleTool(name, args, options = {}) {
  const approved = options.approved === true;

  if (HIGH_RISK_TOOL_NAMES.has(name) && !approved) {
    throw new Error("この操作は承認が必要です");
  }

  switch (name) {
    case "list_files":
      return await listFiles(args.relative_path ?? ".");
    case "read_file":
      return await readFile(args.relative_path);
    case "write_file":
      return await writeFile(args.relative_path, args.content);
    case "edit_file":
      return await editFile(args.relative_path, args.find_text, args.replace_text);
    case "fetch_url":
      return await fetchUrl(args.url);
    case "create_vite_react_app":
      return await createViteReactApp(args.app_name);
    case "web_search":
      return await webSearch(args.query);
    case "npm_install":
      return await npmInstall(args.project_dir);
    case "npm_build":
      return await npmBuild(args.project_dir);
    case "npm_test":
      return await npmTest(args.project_dir);
    case "run_node":
      return await runNode(args.project_dir, args.entry ?? "index.js");
    default:
      throw new Error(`unknown tool: ${name}`);
  }
}

async function classifyMessage(text) {
  if (ruleBasedTaskSignal(text)) {
    const size = estimateTaskSize(text);
    return {
      mode: "agent_task",
      size,
      reason: "rule_based_task_signal"
    };
  }

  const { response } = await callLLM("chat", {
    messages: [
      {
        role: "system",
        content:
          "ユーザーのメッセージを 3分類してください。" +
          "返答は JSON のみ。" +
          '{"mode":"chat|dev_consult|agent_task","size":"small|large","reason":"..."}' +
          " chat は雑談や一般質問。" +
          " dev_consult は要件定義、設計相談、実装相談、レビュー相談。" +
          " agent_task は実際に workspace のファイル変更やコマンド実行が必要な依頼。" +
          " agent_task 以外でファイルやコマンドを使わない。"
      },
      {
        role: "user",
        content: text
      }
    ],
    temperature: 0
  });

  const raw = getMessageText(response.choices?.[0]?.message);
  const parsed = extractJsonObject(raw);

  if (parsed) {
    return {
      mode: parsed.mode || "chat",
      size: parsed.size || "small",
      reason: parsed.reason || ""
    };
  }

  return {
    mode: "chat",
    size: "small",
    reason: "fallback_parse_failed"
  };
}

async function planGoal(goal) {
  const { response } = await callLLM("plan", {
    messages: [
      {
        role: "system",
        content:
          "あなたはDiscord上の開発エージェントです。" +
          "ユーザーの依頼を実行する前に、短い実行計画を作ってください。" +
          "最新の仕様や公開情報が必要な場合は、内蔵の web search / web extractor を使って確認してください。" +
          "JSONだけ返してください。" +
          '{"summary":"...", "plan":["step1","step2","step3"], "notes":"..."}'
      },
      {
        role: "user",
        content: goal
      }
    ],
    temperature: 0.2
  });

  const raw = getMessageText(response.choices?.[0]?.message);
  const parsed = extractJsonObject(raw);

  if (parsed) {
    return parsed;
  }

  return {
    summary: "計画を生成しました",
    plan: [raw || "plan unavailable"],
    notes: ""
  };
}

async function executeGoal(task) {
  const approved = task?.approval_granted === true;
  const tools = getToolsForTask(task);

  const messages = [
    {
      role: "system",
      content:
        "あなたはDiscord上の安全寄りな開発エージェントです。" +
        "workspace配下だけで作業してください。" +
        "必要に応じてツールを使い、アプリ作成・編集・ビルド確認を行ってください。" +
        "最新仕様や外部情報が必要な場合は web_search を使って確認してください。" +
        "最後は日本語で簡潔に、何を作ったか・どこに作ったか・次に何をすべきか報告してください。"
    },
    {
      role: "user",
      content:
        `タスクID: ${task.id}\n` +
        `依頼: ${task.goal}\n` +
        `計画:\n- ${(task.plan || []).join("\n- ")}`
    }
  ];

  for (let i = 0; i < 12; i++) {
    const { route, response } = await callLLM("execute", {
      messages,
      tools,
      temperature: 0.1,
      max_tokens: 1600
    });

    const message = response.choices?.[0]?.message;
    const toolCalls = message?.tool_calls || [];
    const finalText = getMessageText(message);

    if (!toolCalls.length) {
      return finalText || `完了しました。provider=${route.provider}, model=${route.model}`;
    }

    messages.push({
      role: "assistant",
      content: message.content || "",
      tool_calls: toolCalls
    });

    for (const toolCall of toolCalls) {
      const fn = toolCall.function || {};
      const name = fn.name;
      let args = {};

      try {
        args = JSON.parse(fn.arguments || "{}");
      } catch {
        args = {};
      }

      try {
        const result = await handleTool(name, args, { approved });
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify(result, null, 2)
        });
      } catch (err) {
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({ error: String(err.message || err) })
        });
      }
    }
  }

  return "ツール実行の上限に達しました。";
}

async function handleSimpleAI(prompt) {
  const messages = [
    {
      role: "system",
      content:
        "あなたはDiscord上の親切なアシスタントです。日本語で簡潔かつ実用的に答えてください。" +
        "URLが含まれている場合、必要に応じて fetch_url を使って内容を確認してから答えてください。" +
        "GitHub の blob URL が含まれる場合も、必要に応じて fetch_url を使って内容を確認してください。" +
        "最新情報、ニュース、モデルの新情報、価格、API仕様、公開日、バージョンなど、" +
        "変わりうる情報は必要に応じて web_search / web_extractor を使って確認してから答えてください。" +
        "検索結果を使った場合でも、最後は必ずユーザー向けの自然な日本語の文章で答えてください。" +
        "JSONだけを返したり、actionオブジェクトだけを返したりしないでください。"
    },
    {
      role: "user",
      content: prompt
    }
  ];

  for (let i = 0; i < 6; i++) {
    const { response, route } = await callLLM("chat", {
      messages,
      tools: TOOLS.filter(t => SAFE_TOOL_NAMES.has(t.function.name)),
      temperature: 0.4,
      max_tokens: 1400
    });

    console.log("handleSimpleAI raw response:", JSON.stringify(response, null, 2));

    const message = response?.choices?.[0]?.message;
    const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
    let text = getMessageText(message);

    if (!text) {
      text =
        response?.output_text ||
        message?.output_text ||
        message?.text ||
        message?.refusal ||
        "";
    }

    if (!toolCalls.length) {
      const cleaned = String(text || "").trim();

      if (cleaned) {
        return cleaned;
      }

      return `返答を取得できませんでした。provider=${route.provider}, model=${route.model}`;
    }

    messages.push({
      role: "assistant",
      content: text || message?.content || "",
      tool_calls: toolCalls
    });

    for (const toolCall of toolCalls) {
      const fn = toolCall.function || {};
      const name = fn.name;
      let args = {};

      try {
        args = JSON.parse(fn.arguments || "{}");
      } catch (err) {
        console.error("tool args parse error:", err);
        args = {};
      }

      try {
        const result = await handleTool(name, args, { approved: false });

        console.log(
          "handleSimpleAI tool result:",
          JSON.stringify({ name, args, result }, null, 2)
        );

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify(result, null, 2)
        });
      } catch (err) {
        console.error("handleSimpleAI tool error:", err);

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify(
            { error: String(err?.message || err) },
            null,
            2
          )
        });
      }
    }
  }

  return "検索付き応答の上限に達しました。";
}

function formatTask(task) {
  return [
    `ID: ${task.id}`,
    `status: ${task.status}`,
    `goal: ${task.goal}`,
    task.plan?.length ? `plan:\n- ${task.plan.join("\n- ")}` : "",
    task.notes ? `notes: ${task.notes}` : "",
    task.result ? `result:\n${clip(task.result, 700)}` : ""
  ]
    .filter(Boolean)
    .join("\n");
}

async function handleUsage() {
  const u = await readJson(USAGE_FILE, null);
  if (!u) return "usageなし";

  const modelLines = Object.entries(u.by_model || {}).map(([name, stats]) => {
    return `${name}: requests=${stats.requests}, total_tokens=${stats.total_tokens}`;
  });

  return [
    `session started: ${u.started_at}`,
    `requests: ${u.requests}`,
    `input_tokens: ${u.input_tokens}`,
    `output_tokens: ${u.output_tokens}`,
    `total_tokens: ${u.total_tokens}`,
    "",
    ...modelLines
  ].join("\n");
}

function isValidTaskType(taskType) {
  return ["chat", "plan", "execute"].includes(taskType);
}

function parseProviderCommand(content) {
  const parts = String(content || "").trim().split(/\s+/);
  return parts;
}

function providerHelpText() {
  return standardCommandGuide();
}

function getProviderStatusText() {
  return standardProviderStatusText();
}

async function handleProviderCommand(message) {
  const parts = parseProviderCommand(message.content);

  if (parts.length === 1) {
    await message.reply(getProviderStatusText());
    return true;
  }

  if (parts[1] === "help") {
    await message.reply(providerHelpText());
    return true;
  }

  if (parts[1] === "reset") {
    const scope = (parts[2] || "").toLowerCase();

    if (scope === "all") {
      runtimeRouteOverrides.chat = null;
      runtimeRouteOverrides.plan = null;
      runtimeRouteOverrides.execute = null;
      await message.reply("runtime override をすべてリセットしました。\n\n" + getProviderStatusText());
      return true;
    }

    if (!isValidTaskType(scope)) {
      await message.reply("reset 対象は chat / plan / execute / all のいずれかです。\n\n" + providerHelpText());
      return true;
    }

    runtimeRouteOverrides[scope] = null;
    await message.reply(`${scope} の runtime override をリセットしました。\n\n` + getProviderStatusText());
    return true;
  }

  const taskType = (parts[1] || "").toLowerCase();
  const provider = normalizeProviderName(parts[2] || "");
  const model = parts.slice(3).join(" ").trim();

  if (!isValidTaskType(taskType)) {
    await message.reply("対象は chat / plan / execute のいずれかです。\n\n" + providerHelpText());
    return true;
  }

  if (!provider || !model) {
    await message.reply("provider と model を指定してください。\n\n" + providerHelpText());
    return true;
  }

  if (!["openai", "gemini", "anthropic", "qwen"].includes(provider)) {
    await message.reply("provider は openai / gemini / anthropic(claude) / qwen のいずれかです。");
    return true;
  }

  runtimeRouteOverrides[taskType] = {
    provider,
    model
  };

  await message.reply(
    [
      `${taskType} を切り替えました。`,
      `provider: ${provider}`,
      `model: ${model}`,
      "",
      getProviderStatusText()
    ].join("\n")
  );
  return true;
}

async function handleNaturalMessage(message) {
  const cleaned = removeBotMention(message.content);

  if (!cleaned) {
    await message.reply("内容を入れてください。");
    return;
  }

  if (cleaned === "help" || cleaned === "ヘルプ") {
    await message.reply(standardCommandGuide());
    return;
  }

  const cls = await classifyMessage(cleaned);

  if (cls.mode === "chat" || cls.mode === "dev_consult") {
    await message.channel.sendTyping();
    const text = await handleSimpleAI(cleaned);
    await message.reply(clip(text));
    return;
  }

  if (cls.mode === "agent_task") {
    await message.channel.sendTyping();

    const task = await createTask(cleaned, message.author.username, "natural");
    const plan = await planGoal(cleaned);

    const needsApproval = cls.size !== "small" || requiresApprovalByPolicy(cleaned);

    const updated = await updateTask(task.id, {
      plan: Array.isArray(plan.plan) ? plan.plan : [String(plan.plan || "")],
      notes: String(plan.notes || ""),
      status: needsApproval ? "waiting_approval" : "running"
    });

    if (!needsApproval) {
      await message.reply(
        [
          `小規模タスクとして自動実行します。`,
          `taskId: ${updated.id}`,
          "",
          formatTask(updated)
        ].join("\n")
      );

      const result = await executeGoal(updated);
      const done = await updateTask(updated.id, {
        status: "done",
        result
      });

      await message.reply(
        [
          `実行完了: ${done.id}`,
          "",
          clip(done.result)
        ].join("\n")
      );
      return;
    }

    await message.reply(
      [
        `大きめのタスクとして計画を作成しました。`,
        `taskId: ${updated.id}`,
        "",
        formatTask(updated),
        "",
        `実行するなら: !approve ${updated.id}`
      ].join("\n")
    );
    return;
  }

  await message.reply("判定できなかったため、通常応答します。");
  const text = await handleSimpleAI(cleaned);
  await message.reply(clip(text));
}

client.once("clientReady", async () => {
  await ensureFiles();
  console.log(`Discord bot ready: ${client.user.tag}`);
  console.log(`Workspace: ${WORKSPACE_ROOT}`);
  console.log(`Agent channel: #${AGENT_CHANNEL_NAME}`);
  console.log(`Default provider: ${DEFAULT_PROVIDER}`);
  console.log(`Chat route:`, getEffectiveRoute("chat"));
  console.log(`Plan route:`, getEffectiveRoute("plan"));
  console.log(`Execute route:`, getEffectiveRoute("execute"));
});

client.on("messageCreate", async message => {
  try {
    if (message.author.bot) return;

    if (message.content === "!status") {
      const tasks = await readTasks();
      if (tasks.length === 0) {
        await message.reply("タスクはまだありません。");
        return;
      }

      const lines = tasks.slice(0, 10).map(
        t => `${t.id} | ${t.status} | ${t.goal}`
      );
      await message.reply(
        [
          "直近のタスク一覧です。",
          "",
          lines.join("\n")
        ].join("\n")
      );
      return;
    }

    if (message.content.startsWith("!status ")) {
      const id = message.content.slice(8).trim();
      const task = await getTask(id);
      if (!task) {
        await message.reply(`task not found: ${id}`);
        return;
      }
      await message.reply(
        [
          "タスクの詳細です。",
          "",
          clip(formatTask(task))
        ].join("\n")
      );
      return;
    }

    if (message.content.startsWith("!approve ")) {
      const id = message.content.slice(9).trim();
      const task = await getTask(id);
      if (!task) {
        await message.reply(`task not found: ${id}`);
        return;
      }

      if (task.status !== "waiting_approval") {
        await message.reply(`このタスクは承認待ちではありません: ${task.status}`);
        return;
      }

      await updateTask(id, { status: "running", approval_granted: true });
      await message.reply(`実行を開始します: ${id}`);
      await message.channel.sendTyping();

      const runningTask = await getTask(id);
      const result = await executeGoal(runningTask);
      const done = await updateTask(id, {
        status: "done",
        result
      });

      await message.reply(
        [
          `実行完了: ${id}`,
          "",
          clip(done.result)
        ].join("\n")
      );
      return;
    }

    if (message.content.startsWith("!provider")) {
      await handleProviderCommand(message);
      return;
    }

    if (message.content === "!usage") {
      const text = await handleUsage();
      await message.reply(
        "現在の利用状況です。\n\n" +
        text +
        "\n\n正確な請求・残高は各プロバイダの Usage / Billing / Limits を確認してください。"
      );
      return;
    }

    if (!shouldHandleMessage(message)) return;

    await handleNaturalMessage(message);
  } catch (err) {
    console.error("message handler error:", err);
    console.error("message handler error stack:", err?.stack || "(no stack)");
    console.error("message handler error raw:", JSON.stringify({
      message: err?.message,
      name: err?.name,
      status: err?.status,
      code: err?.code,
      type: err?.type,
      param: err?.param
    }, null, 2));
    try {
      await message.reply("エラーが発生しました。ログを確認してください。");
    } catch {}
  }
});

client.login(process.env.DISCORD_TOKEN);
