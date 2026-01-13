(() => {
  function normalizeHeadingText(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function buildTOC() {
    const tocRoot = document.getElementById("left-toc");
    const article = document.querySelector("#main.post article.content");
    if (!tocRoot || !article) return;

    const headings = Array.from(article.querySelectorAll("h2, h3")).filter((h) => !!h.id);
    if (headings.length < 2) return;

    const list = document.createElement("ul");
    list.className = "toc-list";

    for (const heading of headings) {
      const item = document.createElement("li");
      const level = heading.tagName === "H2" ? 2 : 3;
      item.className = `toc-item toc-level-${level}`;

      const link = document.createElement("a");
      link.href = `#${heading.id}`;
      link.textContent = normalizeHeadingText(heading.textContent);
      item.appendChild(link);
      list.appendChild(item);
    }

    tocRoot.innerHTML = "";
    tocRoot.appendChild(list);
    document.body.classList.add("has-toc");
  }

  function asSidenoteHTML(footnoteLi) {
    const clone = footnoteLi.cloneNode(true);
    for (const backref of clone.querySelectorAll(".footnote-backref")) backref.remove();
    for (const para of clone.querySelectorAll("p")) {
      para.innerHTML = para.innerHTML.replace(/\s+$/, "");
    }
    return clone.innerHTML.trim();
  }

  function transformFootnotesToSidenotes() {
    const footnotes = document.querySelector("section.footnotes");
    if (!footnotes) return;

    const noteLis = Array.from(footnotes.querySelectorAll("ol > li[id^='fn:']"));
    if (noteLis.length === 0) return;

    const noteById = new Map();
    for (const li of noteLis) noteById.set(li.id, asSidenoteHTML(li));

    const refs = Array.from(document.querySelectorAll("a.footnote-ref[href^='#fn:']"));
    if (refs.length === 0) return;

    for (const ref of refs) {
      const sup = ref.closest("sup");
      if (!sup) continue;

      const href = ref.getAttribute("href") || "";
      const fnId = href.startsWith("#") ? href.slice(1) : href;
      const noteHTML = noteById.get(fnId);
      if (!noteHTML) continue;

      const noteNumber = (ref.textContent || "").trim();
      const sidenoteId = `sn${noteNumber || fnId.replace("fn:", "")}`;

      const label = document.createElement("label");
      label.className = "sidenote-number";
      label.setAttribute("for", sidenoteId);
      label.textContent = noteNumber || "*";
      label.tabIndex = 0;
      label.setAttribute("role", "button");

      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.className = "margin-toggle";
      toggle.id = sidenoteId;

      label.addEventListener("keydown", (e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault();
        toggle.click();
      });

      const sidenote = document.createElement("span");
      sidenote.className = "sidenote";
      sidenote.innerHTML = noteHTML;

      const fragment = document.createDocumentFragment();
      fragment.appendChild(label);
      fragment.appendChild(toggle);
      fragment.appendChild(sidenote);

      sup.replaceWith(fragment);
    }

    document.body.classList.add("has-sidenotes");
  }

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name);
    const trimmed = (v || "").trim();
    return trimmed.length ? trimmed : fallback;
  }

  function initBackToTop() {
    const link = document.getElementById("back-to-top");
    if (!link) return;
    link.addEventListener("click", (e) => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  function initKeyboardNav() {
    const paginator = document.querySelector(".paginator");
    if (!paginator) return;

    const links = Array.from(paginator.querySelectorAll("a.link"));
    const prev = links[0] || null;
    const next = links.length ? links[links.length - 1] : null;
    const prevUrl = prev && prev.getAttribute("href") !== "#" ? prev.href : null;
    const nextUrl = next && next.getAttribute("href") !== "#" ? next.href : null;

    document.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target && e.target.isContentEditable)) return;

      if (e.key === "ArrowLeft" && prevUrl) {
        e.preventDefault();
        window.location.href = prevUrl;
      }
      if (e.key === "ArrowRight" && nextUrl) {
        e.preventDefault();
        window.location.href = nextUrl;
      }
    });
  }

  function initZoom() {
    const article = document.querySelector("#main.post article.content");
    if (!article) return;

    for (const img of article.querySelectorAll("img")) {
      if (!img.hasAttribute("data-zoomable")) img.setAttribute("data-zoomable", "");
    }

    if (typeof window.mediumZoom === "function") {
      window.mediumZoom("[data-zoomable]", {
        margin: 24,
        background: "rgba(0, 0, 0, 0.92)",
      });
    }
  }

  function initMermaid() {
    if (!window.mermaid) return;

    const codes = Array.from(document.querySelectorAll("pre > code.language-mermaid"));
    if (codes.length === 0) return;

    for (const code of codes) {
      const pre = code.parentElement;
      if (!pre) continue;

      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent || "";
      pre.replaceWith(div);
    }

    try {
      window.mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        themeVariables: {
          fontFamily: cssVar("--font-sans", "sans-serif"),
          primaryColor: cssVar("--accent2", "#E6D4C3"),
          primaryTextColor: cssVar("--fg1", "#000000"),
          primaryBorderColor: cssVar("--accent", "#FF9101"),
          lineColor: cssVar("--fg-muted", "#8a8a8a"),
          background: cssVar("--bg", "#ffffff"),
          mainBkg: cssVar("--bg_code", "#f3f3f2"),
          textColor: cssVar("--fg1", "#000000"),
        },
      });

      window.mermaid.run();
    } catch {
      // Avoid breaking the page if Mermaid fails to parse.
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildTOC();
    transformFootnotesToSidenotes();
    initBackToTop();
    initKeyboardNav();
    initZoom();
    initMermaid();
  });
})();
