import "dotenv/config";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";

const AI_API_URL = process.env.AI_API_URL ?? "https://ai.hackclub.com/proxy/v1/chat/completions";
const AI_MODEL = process.env.AI_MODEL ?? process.env.OPENROUTER_MODEL ?? "openai/gpt-oss-120b";
const SEARCH_API_URL = process.env.SEARCH_API_URL ?? "https://search.hackclub.com/res/v1/web/search";
const WEB_SEARCH_TOOL_NAME = "web_search";
const MAX_TOOL_ROUNDS = 8;
const DEFAULT_CONTEXT_THRESHOLD = 20;
const SYSTEM_PROMPT = [
    "You are a local MCP assistant running on the user machine.",
    "Use MCP tools for browser tasks instead of speculating about unavailable environments.",
    "Do not claim you are in a cloud sandbox or cannot access localhost unless a tool explicitly reports that error.",
    "When a tool fails, explain the exact failure and suggest a concrete local fix.",
];

function isWebSearchEnabled(): boolean {
    return (process.env.WEB_SEARCH_STATUS ?? "true").trim().toLowerCase() !== "false";
}

function parseSystemPrompt(): string {
    const lines = [...SYSTEM_PROMPT];
    if (isWebSearchEnabled()) {
        lines.push("Use the built-in web_search tool for current web information. The time you are getting is in UTC.");
    }
    return lines.join(" ");
}

function parseContextThreshold(): number {
    const raw = process.env.CONTEXT_THRESHOLD;
    if (!raw) {
        return DEFAULT_CONTEXT_THRESHOLD;
    }

    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) {
        return DEFAULT_CONTEXT_THRESHOLD;
    }

    return Math.max(0, parsed);
}

type OpenRouterTool = {
    type: "function";
    function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
    };
};

type OpenRouterToolCall = {
    id: string;
    type: "function";
    function: {
        name: string;
        arguments: string;
    };
};

type OpenRouterMessage =
    | {
        role: "system" | "user";
        content: string;
    }
    | {
        role: "assistant";
        content: string;
        tool_calls?: OpenRouterToolCall[];
    }
    | {
        role: "tool";
        tool_call_id: string;
        name: string;
        content: string;
    };

type OpenRouterResponse = {
    choices?: Array<{
        message?: {
            content?: string | null;
            tool_calls?: OpenRouterToolCall[];
        };
    }>;
};

class MCPClient {
    private mcp: Client;
    private transport: StdioClientTransport | null = null;
    private tools: OpenRouterTool[] = [];
    private readonly contextThreshold: number;
    private conversationTurns: OpenRouterMessage[][] = [];

    constructor() {
        this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
        this.contextThreshold = parseContextThreshold();
    }

    private getBuiltInTools(): OpenRouterTool[] {
        if (!isWebSearchEnabled()) {
            return [];
        }

        return [
            {
                type: "function",
                function: {
                    name: WEB_SEARCH_TOOL_NAME,
                    description: "Search the web for current information.",
                    parameters: {
                        type: "object",
                        properties: {
                            query: {
                                type: "string",
                            },
                        },
                        required: ["query"],
                    },
                },
            },
        ];
    }

    private async callBuiltInTool(args: Record<string, unknown>) {
        const query = String(args.query ?? "").trim();
        const response = await fetch(`${SEARCH_API_URL}?q=${encodeURIComponent(query)}&count=5`, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${process.env.HACK_CLUB_SEARCH_API_KEY ?? ""}`,
            },
        });

        return {
            toolResult: await response.json(),
        };
    }

    async connectToServer(serverScriptPath: string, serverArgs: string[] = []) {
        try {
            const isJs = serverScriptPath.endsWith(".js");
            const isPy = serverScriptPath.endsWith(".py");
            if (!isJs && !isPy) {
                throw new Error("Server script must be a .js or .py file");
            }
            const command = isPy
                ? (process.platform === "win32" ? "python" : "python3")
                : process.execPath;

            this.transport = new StdioClientTransport({ command, args: [serverScriptPath, ...serverArgs] });
            await this.mcp.connect(this.transport);

            const toolsResult = await this.mcp.listTools();
            this.tools = toolsResult.tools.map((tool) => ({
                type: "function",
                function: {
                    name: tool.name,
                    description: tool.description ?? "",
                    parameters: (tool.inputSchema as Record<string, unknown>) ?? {
                        type: "object",
                        properties: {},
                    },
                },
            }));
            this.tools.push(...this.getBuiltInTools());
            console.log("Connected to server with tools:", this.tools.map((tool) => tool.function.name));
        } catch (e) {
            console.log("Failed to connect to MCP server: ", e);
            throw e;
        }
    }

    private async callOpenRouter(messages: OpenRouterMessage[]): Promise<OpenRouterResponse> {
        const apiKey = process.env.HACKCLUB_API_KEY ?? process.env.OPENROUTER_API_KEY;
        if (!apiKey) {
            throw new Error("Missing HACKCLUB_API_KEY or OPENROUTER_API_KEY");
        }

        const headers: Record<string, string> = {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
        };

        if (process.env.OPENROUTER_HTTP_REFERER) {
            headers["HTTP-Referer"] = process.env.OPENROUTER_HTTP_REFERER;
        }
        if (process.env.OPENROUTER_APP_TITLE) {
            headers["X-OpenRouter-Title"] = process.env.OPENROUTER_APP_TITLE;
        }

        const body: Record<string, unknown> = {
            model: AI_MODEL,
            messages,
        };

        if (this.tools.length > 0) {
            body.tools = this.tools;
            body.tool_choice = "auto";
        }

        const response = await fetch(AI_API_URL, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`AI API ${response.status}: ${errorText}`);
        }

        return (await response.json()) as OpenRouterResponse;
    }

    private parseToolArguments(rawArguments: string): Record<string, unknown> {
        if (!rawArguments) {
            return {};
        }

        try {
            const parsed = JSON.parse(rawArguments) as unknown;
            if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
                return parsed as Record<string, unknown>;
            }
        } catch {
            // Empty
        }

        return {};
    }

    private toolResultToText(result: unknown): string {
        if (!result || typeof result !== "object") {
            return "(tool returned no text content)";
        }

        const resultRecord = result as {
            content?: unknown;
            toolResult?: unknown;
        };

        if (Array.isArray(resultRecord.content)) {
            const contentText = resultRecord.content
                .map((block) => {
                    if (
                        block
                        && typeof block === "object"
                        && "type" in block
                        && "text" in block
                        && (block as { type?: unknown }).type === "text"
                        && typeof (block as { text?: unknown }).text === "string"
                    ) {
                        return (block as { text: string }).text;
                    }
                    return JSON.stringify(block);
                })
                .join("\n")
                .trim();

            return contentText || "(tool returned no text content)";
        }

        if (resultRecord.toolResult !== undefined) {
            return JSON.stringify(resultRecord.toolResult);
        }

        return "(tool returned no text content)";
    }

    private buildMessagesForQuery(query: string): OpenRouterMessage[] {
        const messages: OpenRouterMessage[] = [{ role: "system", content: parseSystemPrompt() }];
        messages.push({ role: "system", content: `Current local date/time: ${new Date().toISOString()}` });

        const historyToInclude = this.contextThreshold === 0
            ? []
            : this.conversationTurns.slice(-this.contextThreshold);

        for (const turn of historyToInclude) {
            messages.push(...turn);
        }

        messages.push({ role: "user", content: query });
        return messages;
    }

    private trimConversationTurns() {
        if (this.contextThreshold === 0) {
            this.conversationTurns = [];
            return;
        }

        if (this.conversationTurns.length > this.contextThreshold) {
            this.conversationTurns = this.conversationTurns.slice(-this.contextThreshold);
        }
    }

    async processQuery(query: string) {
        const messages = this.buildMessagesForQuery(query);
        const currentTurn: OpenRouterMessage[] = [{ role: "user", content: query }];
        const finalText: string[] = [];

        for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
            const response = await this.callOpenRouter(messages);
            const assistantMessage = response.choices?.[0]?.message;

            if (!assistantMessage) {
                throw new Error("AI API returned no completion choices");
            }

            const assistantText = (assistantMessage.content ?? "").trim();
            const toolCalls = assistantMessage.tool_calls ?? [];

            if (assistantText) {
                finalText.push(assistantText);
            }

            messages.push({
                role: "assistant",
                content: assistantText,
                ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
            });
            currentTurn.push({
                role: "assistant",
                content: assistantText,
                ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
            });

            if (toolCalls.length === 0) {
                break;
            }

            for (const toolCall of toolCalls) {
                const args = this.parseToolArguments(toolCall.function.arguments);
                const result = toolCall.function.name === WEB_SEARCH_TOOL_NAME
                    ? await this.callBuiltInTool(args)
                    : await this.mcp.callTool({
                        name: toolCall.function.name,
                        arguments: args,
                    });

                messages.push({
                    role: "tool",
                    tool_call_id: toolCall.id,
                    name: toolCall.function.name,
                    content: this.toolResultToText(result),
                });
                currentTurn.push({
                    role: "tool",
                    tool_call_id: toolCall.id,
                    name: toolCall.function.name,
                    content: this.toolResultToText(result),
                });
            }
        }

        this.conversationTurns.push(currentTurn);
        this.trimConversationTurns();

        return finalText.join("\n") || "(no text response)";
    }

    async chatLoop() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        try {
            console.log("\nMCP Client Started!");
            console.log("Type your queries or 'quit' to exit.");
            console.log(`Context threshold: ${this.contextThreshold} prompt(s)`);

            while (true) {
                let message: string;
                try {
                    message = await rl.question("\nQuery: ");
                } catch (error) {
                    const errorCode = (error as { code?: string }).code;
                    if (errorCode === "ERR_USE_AFTER_CLOSE") {
                        break;
                    }
                    throw error;
                }

                if (message.toLowerCase() === "quit") {
                    break;
                }
                const response = await this.processQuery(message);
                console.log("\n" + response);
            }
        } finally {
            rl.close();
        }
    }

    async cleanup() {
        await this.mcp.close();
    }
}

async function main() {
    if (process.argv.length < 3) {
        console.log("Usage: node build/index.js <path_to_server_script> [server_args...]");
        console.log("Example: node build/index.js /path/to/chrome-devtools-mcp.js --browser-url=http://127.0.0.1:9222");
        return;
    }

    const serverScriptPath = process.argv[2];
    const serverArgs = process.argv.slice(3);
    const mcpClient = new MCPClient();
    try {
        await mcpClient.connectToServer(serverScriptPath, serverArgs);

        const apiKey = process.env.HACKCLUB_API_KEY ?? process.env.OPENROUTER_API_KEY;
        if (!apiKey) {
            console.log(
                "\nNo API key found. To query these tools, set one of:"
                + "\n  export HACKCLUB_API_KEY=your-api-key-here"
                + "\n  export OPENROUTER_API_KEY=your-api-key-here"
            );
            return;
        }

        await mcpClient.chatLoop();
    } catch (e) {
        console.error("Error:", e);
        process.exit(1);
    } finally {
        await mcpClient.cleanup();
        process.exit(0);
    }
}

main();