const P: Record<string, string> = {
  play: "M5 3.5v13l11-6.5z",
  pause: "M5 3.5h3.5v13H5zM11.5 3.5H15v13h-3.5z",
  next: "M4 4l8 6-8 6zM13.5 4H16v12h-2.5z",
  prev: "M16 4l-8 6 8 6zM4 4h2.5v12H4z",
  close: "M5.1 4L10 8.9 14.9 4 16 5.1 11.1 10 16 14.9 14.9 16 10 11.1 5.1 16 4 14.9 8.9 10 4 5.1z",
};

export function Icon({ name, size = 18 }: { name: keyof typeof P | string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" focusable="false">
      <path d={P[name]} fill="currentColor" />
    </svg>
  );
}

/** The wordmark's route: a seed, three steps down, an end in the tail. */
export function Mark() {
  return (
    <svg width="26" height="18" viewBox="0 0 26 18" aria-hidden="true">
      <path d="M3 3 L9 6 L15 9 L23 15" stroke="var(--accent)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <circle cx="3" cy="3" r="2.5" fill="var(--ink)" />
      <circle cx="23" cy="15" r="2.8" fill="var(--accent)" />
    </svg>
  );
}
