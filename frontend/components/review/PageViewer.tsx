"use client";

import clsx from "clsx";
import type { FieldOut, PageOut } from "@/lib/api";
import { label } from "@/lib/format";

/** The page as photographed, with every extracted value boxed where Keel read it. */
export function PageViewer({
  documentId,
  page,
  fields,
  activeId,
  onSelect,
  lowConfidence = 0.75,
}: {
  documentId: string;
  page: PageOut;
  fields: FieldOut[];
  activeId: string | null;
  onSelect: (id: string) => void;
  lowConfidence?: number;
}) {
  const boxed = fields.filter((f) => f.box && f.box.page_id === page.id);
  return (
    <figure className="relative overflow-hidden rounded-xl border border-line bg-white shadow-sm">
      {/* eslint-disable-next-line @next/next/no-img-element -- authenticated, same-origin page render */}
      <img
        src={`/api/documents/${documentId}/pages/${page.n}/image`}
        alt={`Page ${page.n}`}
        width={page.width}
        height={page.height}
        className="block h-auto w-full select-none"
        draggable={false}
      />
      {boxed.map((f) => {
        const b = f.box!;
        const low = f.status !== "corrected" && f.confidence < lowConfidence;
        const active = f.id === activeId;
        return (
          <button
            key={f.id}
            type="button"
            title={`${label(f.path)}: ${String(f.value ?? "")}`}
            aria-label={`${label(f.path)}: ${String(f.value ?? "unreadable")}`}
            onClick={() => onSelect(f.id)}
            style={{
              left: `calc(${b.x * 100}% - 2px)`,
              top: `calc(${b.y * 100}% - 2px)`,
              width: `calc(${Math.max(b.w * 100, 1.2)}% + 4px)`,
              height: `calc(${Math.max(b.h * 100, 1.2)}% + 4px)`,
            }}
            className={clsx(
              "group absolute rounded-[5px] border transition-[background-color,border-color,box-shadow] duration-150",
              f.status === "corrected" && "border-ledger/60 bg-ledger/[0.07] hover:border-ledger hover:bg-ledger/15",
              f.status !== "corrected" &&
                !low &&
                "border-inkblue/40 bg-inkblue/[0.05] hover:border-inkblue/80 hover:bg-inkblue/[0.12]",
              low && "border-dashed border-carbon bg-carbon/[0.12] hover:bg-carbon/25",
              active &&
                "z-10 border-solid border-inkblue bg-inkblue/10 shadow-[0_0_0_3px_color-mix(in_srgb,var(--ink-blue)_22%,transparent),0_4px_14px_-4px_color-mix(in_srgb,var(--ink-blue)_55%,transparent)]",
            )}
          >
            <span
              className={clsx(
                "pointer-events-none absolute left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium leading-none",
                "bg-ink text-bg shadow-sm transition-opacity duration-150",
                b.y < 0.05 ? "top-full mt-1" : "bottom-full mb-1",
                active ? "opacity-100" : "opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100",
              )}
            >
              {label(f.path)}
              {low && " · check"}
            </span>
          </button>
        );
      })}
      <figcaption className="absolute bottom-2 right-2 rounded-md bg-ink/75 px-2 py-0.5 text-xs text-bg">
        Page {page.n} · {page.has_text_layer ? "text layer" : "read by OCR"}
      </figcaption>
    </figure>
  );
}
