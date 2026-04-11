import { createSignal, For, Show, Match, Switch } from "solid-js";
import { Backend } from "../backend.tsx";

type Tab = "values" | "settings" | "nav";

interface Props {
    tab: Tab;
    backend: Backend;
    onClose: () => void;
}

function ValuesView(props: { backend: Backend }) {
    const [editingName, setEditingName] = createSignal<string | null>(null);
    const [editValue, setEditValue] = createSignal("");
    const [newName, setNewName] = createSignal("");
    const [newValue, setNewValue] = createSignal("");

    return (
        <div class="flex flex-col gap-1.5 p-3">
            <For each={props.backend.values[0]}>
                {(item) => (
                    <div class="rounded bg-stone-800 px-3 py-2.5">
                        <Show
                            when={editingName() === item.name}
                            fallback={
                                <div class="flex items-start justify-between gap-3">
                                    <div class="min-w-0 flex-1">
                                        <div class="text-amber-400 text-sm font-mono truncate">
                                            {props.backend.settings[0].placeholder_prefix}{item.name}
                                        </div>
                                        <div class="text-stone-500 text-xs font-mono mt-1 break-all leading-relaxed">
                                            {item.value.slice(0, 80)}{item.value.length > 80 ? "…" : ""}
                                        </div>
                                    </div>
                                    <div class="flex flex-col gap-0.5 shrink-0">
                                        <button
                                            onClick={() => { setEditingName(item.name); setEditValue(item.name); }}
                                            class="w-5 h-5 flex items-center justify-center rounded
                                                   text-stone-600 hover:text-stone-200 hover:bg-stone-700
                                                   text-xs transition-colors"
                                            title="Rename"
                                        >
                                            ✎
                                        </button>
                                        <button
                                            onClick={() => props.backend.deleteValue(item.name)}
                                            class="w-5 h-5 flex items-center justify-center rounded
                                                   text-stone-600 hover:text-red-400 hover:bg-stone-700
                                                   text-xs transition-colors"
                                            title="Delete"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                </div>
                            }
                        >
                            <input
                                class="bg-stone-700 text-stone-200 text-sm font-mono
                                       rounded px-2.5 py-1.5 outline-none w-full"
                                value={editValue()}
                                onInput={(e) => setEditValue(e.currentTarget.value)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                        props.backend.renameValue(item.name, editValue());
                                        setEditingName(null);
                                    }
                                    if (e.key === "Escape") setEditingName(null);
                                }}
                                autofocus
                            />
                        </Show>
                    </div>
                )}
            </For>

            <Show when={props.backend.values[0].length === 0}>
                <div class="text-stone-600 text-sm italic px-3 py-2">no saved values</div>
            </Show>

            <div class="flex flex-col gap-2 mt-2 px-0">
                <input
                    placeholder="name"
                    class="bg-stone-800 text-stone-200 text-sm font-mono rounded px-3 py-2
                           outline-none placeholder:text-stone-600"
                    value={newName()}
                    onInput={(e) => setNewName(e.currentTarget.value)}
                />
                <input
                    placeholder="value"
                    class="bg-stone-800 text-stone-200 text-sm font-mono rounded px-3 py-2
                           outline-none placeholder:text-stone-600"
                    value={newValue()}
                    onInput={(e) => setNewValue(e.currentTarget.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter" && newName().trim()) {
                            props.backend.saveValue(newName().trim(), newValue());
                            setNewName(""); setNewValue("");
                        }
                    }}
                />
                <button
                    onClick={() => {
                        if (!newName().trim()) return;
                        props.backend.saveValue(newName().trim(), newValue());
                        setNewName(""); setNewValue("");
                    }}
                    class="text-sm text-stone-500 hover:text-amber-400 text-left px-1 py-1 transition-colors"
                >
                    + add
                </button>
            </div>
        </div>
    );
}

type ConnStatus = "idle" | "connecting" | "ok" | "error";

function SettingsView(props: { backend: Backend }) {
    const cfg = () => props.backend.settings[0];
    const [connStatus, setConnStatus] = createSignal<ConnStatus>("idle");
    const [connError, setConnError] = createSignal("");

    const save = (patch: Parameters<typeof props.backend.updateSettings>[0]) =>
        props.backend.updateSettings(patch);

    const connect = async () => {
        setConnStatus("connecting");
        setConnError("");
        const err = await props.backend.refreshSchema();
        if (err === null) {
            setConnStatus("ok");
        } else {
            setConnStatus("error");
            setConnError(err);
        }
    };

    const inputCls = "bg-stone-800 text-stone-200 text-sm font-mono rounded px-3 py-2 outline-none w-full";
    const labelCls = "text-stone-500 text-xs mb-1";

    return (
        <div class="flex flex-col gap-5 p-3">

            {/* Connection */}
            <div class="flex flex-col gap-3">
                <div class="text-stone-400 text-xs uppercase tracking-widest">Connection</div>

                <div class="flex flex-col gap-1">
                    <div class={labelCls}>Endpoint</div>
                    <input
                        class={inputCls}
                        value={cfg().surreal_endpoint}
                        placeholder="ws://localhost:8000"
                        onBlur={(e) => save({ surreal_endpoint: e.currentTarget.value })}
                        onKeyDown={(e) => e.key === "Enter" && save({ surreal_endpoint: (e.target as HTMLInputElement).value })}
                    />
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div class="flex flex-col gap-1">
                        <div class={labelCls}>Namespace</div>
                        <input
                            class={inputCls}
                            value={cfg().surreal_namespace}
                            placeholder="test"
                            onBlur={(e) => save({ surreal_namespace: e.currentTarget.value })}
                            onKeyDown={(e) => e.key === "Enter" && save({ surreal_namespace: (e.target as HTMLInputElement).value })}
                        />
                    </div>
                    <div class="flex flex-col gap-1">
                        <div class={labelCls}>Database</div>
                        <input
                            class={inputCls}
                            value={cfg().surreal_database}
                            placeholder="test"
                            onBlur={(e) => save({ surreal_database: e.currentTarget.value })}
                            onKeyDown={(e) => e.key === "Enter" && save({ surreal_database: (e.target as HTMLInputElement).value })}
                        />
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div class="flex flex-col gap-1">
                        <div class={labelCls}>Username</div>
                        <input
                            class={inputCls}
                            value={cfg().surreal_username}
                            placeholder="root"
                            onBlur={(e) => save({ surreal_username: e.currentTarget.value })}
                            onKeyDown={(e) => e.key === "Enter" && save({ surreal_username: (e.target as HTMLInputElement).value })}
                        />
                    </div>
                    <div class="flex flex-col gap-1">
                        <div class={labelCls}>Password</div>
                        <input
                            class={inputCls}
                            type="password"
                            value={cfg().surreal_password}
                            placeholder="root"
                            onBlur={(e) => save({ surreal_password: e.currentTarget.value })}
                            onKeyDown={(e) => e.key === "Enter" && save({ surreal_password: (e.target as HTMLInputElement).value })}
                        />
                    </div>
                </div>

                <div class="flex items-center gap-3 mt-1">
                    <button
                        onClick={connect}
                        disabled={connStatus() === "connecting"}
                        class="px-4 py-1.5 rounded bg-stone-700 text-stone-200 text-sm
                               hover:bg-stone-600 disabled:opacity-40 disabled:cursor-not-allowed
                               transition-colors"
                    >
                        {connStatus() === "connecting" ? "Connecting…" : "Connect"}
                    </button>
                    <Switch>
                        <Match when={connStatus() === "ok"}>
                            <span class="text-emerald-400 text-xs">Schema loaded</span>
                        </Match>
                        <Match when={connStatus() === "error"}>
                            <span class="text-red-400 text-xs break-all">
                                {connError()}
                            </span>
                        </Match>
                    </Switch>
                </div>
            </div>

            {/* App */}
            <div class="flex flex-col gap-3">
                <div class="text-stone-400 text-xs uppercase tracking-widest">App</div>
                <div class="flex flex-col gap-1">
                    <div class={labelCls}>Placeholder prefix</div>
                    <div class="flex items-center gap-3">
                        <input
                            class="bg-stone-800 text-stone-200 text-sm font-mono rounded px-3 py-2 outline-none w-20"
                            value={cfg().placeholder_prefix}
                            maxLength={4}
                            onBlur={(e) => save({ placeholder_prefix: e.currentTarget.value })}
                            onKeyDown={(e) => e.key === "Enter" && save({ placeholder_prefix: (e.target as HTMLInputElement).value })}
                        />
                        <span class="text-stone-600 text-xs">
                            e.g. <span class="text-amber-400 font-mono">{cfg().placeholder_prefix}last-id</span>
                        </span>
                    </div>
                </div>
            </div>

        </div>
    );
}

function resultMeta(result: string | null): string {
    if (!result) return "—";
    try {
        const p = JSON.parse(result);
        if (Array.isArray(p)) return `array [${p.length}]`;
        if (typeof p === "object" && p !== null) return `object {${Object.keys(p).length}}`;
        if (typeof p === "string") return `string`;
        if (typeof p === "number") return String(p);
        if (typeof p === "boolean") return String(p);
        return typeof p;
    } catch { return "error"; }
}

function NavView(props: { backend: Backend }) {
    const done = () => props.backend.blocks[0].filter(b => b.state === "Done" && b.query.trim());

    return (
        <div class="flex flex-col gap-1 p-3">
            <For each={done()}>
                {(block) => (
                    <button
                        onClick={() => {
                            const scroller = document.getElementById("scroll-root");
                            const el = document.getElementById(`block-${block.id}`);
                            if (el && scroller) {
                                const top = el.offsetTop - 12;
                                scroller.scrollTo({ top, behavior: "smooth" });
                            }
                        }}
                        class="text-left rounded bg-stone-800 px-3 py-2 hover:bg-stone-700
                               transition-colors group"
                    >
                        <div class="text-stone-300 text-sm font-mono truncate">
                            {block.query.length > 60 ? block.query.slice(0, 60) + "…" : block.query}
                        </div>
                        <div class="text-stone-600 text-xs font-mono mt-0.5">
                            {resultMeta(block.result)}
                        </div>
                    </button>
                )}
            </For>
            <Show when={done().length === 0}>
                <div class="text-stone-600 text-sm italic px-3 py-2">no executed queries</div>
            </Show>
        </div>
    );
}

export function Sidebar(props: Props) {
    return (
        <div class="w-1/3 min-w-72 bg-stone-900 flex flex-col overflow-hidden shrink-0">
            <div class="flex-1 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                <Show when={props.tab === "values"}>
                    <ValuesView backend={props.backend} />
                </Show>
                <Show when={props.tab === "settings"}>
                    <SettingsView backend={props.backend} />
                </Show>
                <Show when={props.tab === "nav"}>
                    <NavView backend={props.backend} />
                </Show>
            </div>
        </div>
    );
}
