import "./App.css";
import { QueryBar } from "./components/QueryBar";
import * as Icon from "./components/Feather";

export default function App() {
  return (
    <main class="relative h-screen w-screen bg-stone-800 flex flex-col overflow-hidden">

      {/* Canvas — output blocks */}
      <div class="flex-1 overflow-hidden" />

      {/* Query row */}
      <div class="flex items-end gap-3 p-4">
        <QueryBar />
        <button
          class="shrink-0 w-12 h-12 rounded-full flex items-center justify-center focus:outline-none"
          style={{ background: "#e8b87d" }}
        >
          <Icon.Terminal size={20} stroke="#1a1a1a" strokeWidth={2.2} />
        </button>
      </div>

    </main>
  );
}
