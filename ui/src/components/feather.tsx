import {JSX} from "solid-js";

interface Props {
    size?: number;
    stroke?: string;
    strokeWidth?: number;
    class?: string;
    style?: JSX.CSSProperties;
}

function icon(paths: () => JSX.Element) {
    return (props: Props) => (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            width={props.size ?? 24}
            height={props.size ?? 24}
            fill="none"
            stroke={props.stroke ?? "currentColor"}
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width={props.strokeWidth ?? 2}
            class={props.class}
            style={props.style}
        >
            {paths()}
        </svg>
    );
}

export const Terminal = icon(() => (
    <>
        <polyline points="4 17 10 11 4 5"/>
        <line x1="12" x2="20" y1="19" y2="19"/>
    </>
));

export const Settings = icon(() => (
    <>
        <circle cx="12" cy="12" r="3"/>
        <path
            d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </>
));

export const ChevronDown = icon(() => (
    <polyline points="6 9 12 15 18 9"/>
));

export const X = icon(() => (
    <>
        <line x1="18" y1="6" x2="6" y2="18"/>
        <line x1="6" y1="6" x2="18" y2="18"/>
    </>
));

export const ChevronsRight = icon(() => (
    <>
        <polyline points="13 17 18 12 13 7" />
        <polyline points="6 17 11 12 6 7" />
    </>
));

export const CornerDownLeft = icon(() => (
    <>
        <polyline points="9 10 4 15 9 20"/>
        <path d="M20 4v7a4 4 0 0 1-4 4H4"/>
    </>
));

export const ArrowUp = icon(() => (
    <>
        <line x1="12" y1="19" x2="12" y2="5"/>
        <polyline points="5 12 12 5 19 12"/>
    </>
));

export const ArrowDown = icon(() => (
    <>
        <line x1="12" y1="5" x2="12" y2="19"/>
        <polyline points="19 12 12 19 5 12"/>
    </>
));

// ⇧ Shift — outlined upward arrow with stem
export const ShiftKey = icon(() => (
    <>
        <polyline points="5 12 12 4 19 12"/>
        <polyline points="9 12 9 20 15 20 15 12"/>
    </>
));

// ⇥ Tab — left bar + right-pointing arrow
export const TabKey = icon(() => (
    <>
        <line x1="4" y1="5" x2="4" y2="19"/>
        <line x1="4" y1="12" x2="20" y2="12"/>
        <polyline points="14 6 20 12 14 18"/>
    </>
));

// ⌥ Option/Alt symbol
export const Option = icon(() => (
    <>
        <line x1="4" y1="8" x2="11" y2="8"/>
        <line x1="11" y1="8" x2="20" y2="17"/>
        <line x1="15" y1="17" x2="20" y2="17"/>
        <line x1="15" y1="8" x2="20" y2="8"/>
    </>
));

// Navigation marker / jump-to
export const Navigation2 = icon(() => (
    <polygon points="12 2 19 21 12 17 5 21 12 2"/>
));

export const Database = icon(() => (
    <>
        <ellipse cx="12" cy="5" rx="9" ry="3"/>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </>
));

export const List = icon(() => (
    <>
        <line x1="8" x2="21" y1="6" y2="6"/>
        <line x1="8" x2="21" y1="12" y2="12"/>
        <line x1="8" x2="21" y1="18" y2="18"/>
        <line x1="3" x2="3.01" y1="6" y2="6"/>
        <line x1="3" x2="3.01" y1="12" y2="12"/>
        <line x1="3" x2="3.01" y1="18" y2="18"/>
    </>
));
