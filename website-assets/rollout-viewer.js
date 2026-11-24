(() => {
  "use strict";

  const status = document.getElementById("trace-status");
  const source = document.getElementById("trace-data");
  if (!status || !source) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const POLICY = {dt: "DT", ppo: "PPO", replay: "Dispatch replay"};
  const METRIC = {
    dispatch: {unit: "MW", note: "Positive values charge the battery; negative values discharge it. Hover the line to inspect signed half-hour means."},
    soc: {unit: "MWh", note: "Stored energy is sampled at the end of each half-hour bin."},
    price: {unit: "A$/MWh", note: "All policies face the same historical spot-price path on a matched month."},
    raiseBid: {unit: "%", note: "Mean raise regulation FCAS bid as a percentage of maximum power; this is offered capacity, not delivered response."},
    lowerBid: {unit: "%", note: "Mean lower regulation FCAS bid as a percentage of maximum power; this is offered capacity, not delivered response."},
    profit: {unit: "A$", note: "Cumulative net energy and FCAS revenue less modelled battery wear."},
    energy: {unit: "A$", note: "Cumulative net energy revenue, including the cost of charging."},
    fcas: {unit: "A$", note: "Cumulative revenue across all eight modelled FCAS services."},
    wear: {unit: "A$", note: "Cumulative modelled degradation cost, shown as a positive cost."},
  };
  const monthFormat = new Intl.DateTimeFormat("en-AU", {month: "long", year: "numeric", timeZone: "UTC"});
  const precise = new Intl.NumberFormat("en-AU", {maximumFractionDigits: 2});
  const currency = new Intl.NumberFormat("en-AU", {style: "currency", currency: "AUD", currencyDisplay: "narrowSymbol", maximumFractionDigits: 0});
  const compact = new Intl.NumberFormat("en-AU", {notation: "compact", maximumFractionDigits: 2});

  function svgNode(tag, attrs = {}, text = "") {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== "") node.textContent = text;
    return node;
  }

  function quantile(values, q) {
    const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const i = (sorted.length - 1) * q;
    const lower = Math.floor(i);
    const upper = Math.ceil(i);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (i - lower);
  }

  function makeNiceRange(values, mode, metric, capacityMwh, powerMw) {
    if (mode === "physical") {
      if (metric === "dispatch") return {min: -powerMw, max: powerMw, step: powerMw / 3};
      if (metric === "soc") return {min: 0, max: capacityMwh, step: capacityMwh / 4};
      if (metric === "raiseBid" || metric === "lowerBid") return {min: 0, max: 100, step: 25};
    }
    let lo, hi, clipped = 0;
    if (mode === "robust") {
      lo = quantile(values, 0.02);
      hi = quantile(values, 0.98);
      clipped = values.filter(value => value < lo || value > hi).length;
    } else {
      lo = Math.min(...values);
      hi = Math.max(...values);
    }
    if (lo === hi) {
      const pad = Math.max(1, Math.abs(lo) * 0.08);
      lo -= pad;
      hi += pad;
    } else {
      const pad = (hi - lo) * 0.09;
      lo -= pad;
      hi += pad;
    }
    const target = 5;
    const rawStep = (hi - lo) / target;
    const power = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1, 2, 2.5, 5, 10].map(factor => factor * power).find(value => value >= rawStep) || power * 10;
    return {
      min: Math.floor(lo / step) * step,
      max: Math.ceil(hi / step) * step,
      step,
      clipped,
      percentile: mode === "robust",
    };
  }

  function start(data) {
    const byId = id => document.getElementById(`trace-${id}`);
    const panel = byId("viewer");
    const month = byId("month");
    const metric = byId("metric");
    const windowSelect = byId("window");
    const scaleSelect = byId("scale");
    const scaleNote = byId("scale-note");
    const checks = byId("policies");
    const scrub = byId("time");
    const chart = byId("chart");
    const frame = byId("frame");
    const reading = byId("reading");
    const note = byId("note");
    const tooltip = byId("tooltip");
    const {episodes, binMinutes, capacityMwh, powerMw} = data;
    const binsPerDay = 24 * 60 / binMinutes;
    const bins = episodes[0].price.length;
    const days = bins / binsPerDay;
    let cursor = 0;
    let plot;
    let pointerActive = false;

    chart.setAttribute("tabindex", "0");
    chart.setAttribute("aria-describedby", "trace-note trace-reading");
    episodes.forEach((episode, index) => {
      month.add(new Option(monthFormat.format(new Date(`${episode.month}-01T00:00:00Z`)), index));
    });
    for (let day = 0; day < days; day += 1) windowSelect.add(new Option(`Day ${day + 1} only`, day));

    function formatValue(value, unit) {
      if (unit === "A$") return currency.format(value);
      if (unit === "%") return `${precise.format(value)}%`;
      return `${precise.format(value)} ${unit}`;
    }

    function labelFor(index) {
      const episode = episodes[Number(month.value)];
      const day = Math.floor(index / binsPerDay) + 1;
      const minutes = (index % binsPerDay) * binMinutes;
      const time = [Math.floor(minutes / 60), minutes % 60].map(n => String(n).padStart(2, "0")).join(":");
      return `${monthFormat.format(new Date(`${episode.month}-01T00:00:00Z`))} · Day ${day}, ${time}`;
    }

    function positionTooltip(xValue, yValues) {
      const width = frame.clientWidth;
      const x = xValue;
      const preferredY = Math.min(...yValues, plot.bottom - 20);
      tooltip.hidden = false;
      const tooltipWidth = tooltip.offsetWidth || 190;
      const tooltipHeight = tooltip.offsetHeight || 120;
      const left = x + tooltipWidth + 24 > width ? Math.max(8, x - tooltipWidth - 14) : Math.max(8, x + 14);
      const top = Math.min(Math.max(8, preferredY - tooltipHeight / 2), frame.clientHeight - tooltipHeight - 8);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    }

    function showBin(index, keepTooltip = true) {
      if (!plot) return;
      cursor = Math.max(plot.start, Math.min(plot.end - 1, Math.round(index)));
      scrub.value = cursor - plot.start;
      const label = labelFor(cursor);
      scrub.setAttribute("aria-valuetext", label);
      plot.marker.replaceChildren(svgNode("line", {
        class: "cursor", x1: plot.x(cursor), x2: plot.x(cursor), y1: plot.top, y2: plot.bottom,
      }));

      const tooltipHeading = document.createElement("strong");
      tooltipHeading.textContent = `${label} · ${plot.unit}`;
      tooltip.replaceChildren(tooltipHeading);
      const entries = document.createElement("div");
      entries.className = "trace-tooltip-row";
      const screenEntries = [label, plot.unit];
      const py = [];
      for (const line of plot.lines) {
        const value = line.values[cursor];
        const rawY = plot.y(value);
        const y = Math.max(plot.top + 5, Math.min(plot.bottom - 5, rawY));
        py.push(y);
        plot.marker.append(svgNode("circle", {
          class: `point trace-${line.key}${y !== rawY ? " clipped" : ""}`,
          cx: plot.x(cursor), cy: y, r: 4.5,
        }));
        const name = document.createElement("span");
        name.className = `tooltip-name trace-${line.key}`;
        name.style.color = `var(--trace)`;
        name.textContent = line.name;
        const number = document.createElement("span");
        number.className = "tooltip-value";
        number.textContent = formatValue(value, plot.unit);
        entries.append(name, number);
        screenEntries.push(`${line.name}: ${number.textContent}`);
      }
      tooltip.append(entries);
      reading.textContent = screenEntries.join(". ");
      if (keepTooltip && plot.lines.length) positionTooltip(plot.x(cursor), py);
      else tooltip.hidden = true;
    }

    function render() {
      const episode = episodes[Number(month.value)];
      const selected = metric.value;
      const {unit, note: explanation} = METRIC[selected];
      const spot = selected === "price";
      checks.disabled = spot;
      const physicalOption = scaleSelect.querySelector('[value="physical"]');
      physicalOption.disabled = !["dispatch", "soc", "raiseBid", "lowerBid"].includes(selected);
      if (physicalOption.disabled && scaleSelect.value === "physical") scaleSelect.value = "exact";
      const lines = spot ? [{key: "price", name: "Spot price", values: episode.price}] :
        Array.from(checks.querySelectorAll("input:checked"), ({value: key}) => ({
          key, name: POLICY[key], values: episode.policies[key][selected].map(value => unit === "%" ? value * 100 : value),
        }));
      const start = windowSelect.value === "all" ? 0 : Number(windowSelect.value) * binsPerDay;
      const end = windowSelect.value === "all" ? bins : start + binsPerDay;
      const visible = lines.flatMap(line => line.values.slice(start, end));
      const mode = scaleSelect.value;
      const range = makeNiceRange(visible.length ? visible : [0, 1], mode, selected, capacityMwh, powerMw);
      const {min, max, step} = range;
      const width = Math.max(260, frame.clientWidth);
      const height = chart.clientHeight || 352;
      const left = width < 500 ? 52 : 62;
      const right = width - 18;
      const top = 30;
      const bottom = height - 36;
      const x = index => left + (index - start + .5) / (end - start) * (right - left);
      const y = value => bottom - (value - min) / (max - min) * (bottom - top);
      chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
      chart.replaceChildren(svgNode("title", {id: "trace-chart-title"},
        `${metric.selectedOptions[0].textContent} · ${month.selectedOptions[0].textContent} · ${windowSelect.selectedOptions[0].textContent}`));
      chart.append(svgNode("text", {x: left, y: 17}, `${metric.selectedOptions[0].textContent} · ${unit}`));
      for (let tick = min, count = 0; tick <= max + step / 100 && count < 18; tick += step, count += 1) {
        const py = y(tick);
        chart.append(svgNode("line", {class: Math.abs(tick) < step / 100 ? "axis" : "gridline", x1: left, x2: right, y1: py, y2: py}));
        chart.append(svgNode("text", {x: left - 7, y: py + 4, "text-anchor": "end"}, compact.format(tick)));
      }
      const tickCount = width > 800 ? 8 : width > 500 ? 5 : 3;
      for (let index = 0; index <= tickCount; index += 1) {
        const fraction = index / tickCount;
        const px = left + fraction * (right - left);
        const hours = (start + fraction * (end - start)) * binMinutes / 60;
        const dayHours = hours % 24;
        const label = windowSelect.value === "all" ? `d${Math.floor(hours / 24) + 1} ${dayHours}h` : `${dayHours}h`;
        chart.append(svgNode("text", {x: px, y: bottom + 21, "text-anchor": index === tickCount ? "end" : index === 0 ? "start" : "middle"}, label));
      }
      for (const line of lines) {
        const points = line.values.slice(start, end).map((value, i) => `${x(start + i).toFixed(2)},${y(value).toFixed(2)}`);
        chart.append(svgNode("polyline", {class: `path trace-${line.key}`, points: points.join(" ")}));
      }
      const marker = svgNode("g");
      chart.append(marker);
      plot = {x, y, min, max, top, bottom, left, right, width, height, start, end, lines, marker, unit};
      scrub.max = end - start - 1;
      const clippedNote = mode === "robust" && range.clipped ? ` ${range.clipped} extreme points extend beyond the view; their exact values remain available in the tooltip.` : "";
      scaleNote.textContent = mode === "robust" ? `Y range ${formatValue(min, unit)} to ${formatValue(max, unit)}${range.clipped ? ` · ${range.clipped} clipped` : ""}` :
        mode === "exact" ? "Y-axis fits all visible values" : "Y-axis uses physical limits";
      note.textContent = lines.length ? explanation + (mode === "robust" ? " Y-axis focuses on the central 96% and adds padding." : "") + clippedNote :
        "Select a policy to display its trace.";
      showBin(cursor, pointerActive || document.activeElement === chart || document.activeElement === scrub);
    }

    month.addEventListener("change", render);
    metric.addEventListener("change", render);
    scaleSelect.addEventListener("change", render);
    windowSelect.addEventListener("change", () => {
      cursor = windowSelect.value === "all" ? 0 : Number(windowSelect.value) * binsPerDay;
      render();
    });
    checks.addEventListener("change", render);
    scrub.addEventListener("focus", () => showBin(cursor));
    scrub.addEventListener("input", () => showBin(plot.start + Number(scrub.value)));
    function pickBin(event) {
      if (!plot) return;
      const bounds = chart.getBoundingClientRect();
      const px = (event.clientX - bounds.left) * plot.width / bounds.width;
      const fraction = (px - plot.left) / (plot.right - plot.left);
      showBin(plot.start + Math.floor(fraction * (plot.end - plot.start)));
    }
    chart.addEventListener("click", pickBin);
    chart.addEventListener("pointerenter", () => { pointerActive = true; });
    chart.addEventListener("pointermove", event => {
      if (event.pointerType === "mouse" || event.pointerType === "pen") pickBin(event);
    });
    chart.addEventListener("pointerleave", () => {
      pointerActive = false;
      if (document.activeElement !== scrub && document.activeElement !== chart) tooltip.hidden = true;
    });
    chart.addEventListener("focus", () => showBin(cursor));
    chart.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? plot.start : event.key === "End" ? plot.end - 1 :
        cursor + (event.key === "ArrowRight" ? 1 : -1);
      showBin(next);
    });
    panel.hidden = false;
    render();
    if ("ResizeObserver" in window) new ResizeObserver(render).observe(frame);
    status.hidden = true;
  }

  fetch(source.href)
    .then(response => {
      if (!response.ok) throw new Error(`Trace request returned ${response.status}`);
      return response.json();
    })
    .then(start)
    .catch(error => {
      status.textContent = `The saved rollout viewer could not load: ${error.message}`;
      console.warn(error);
    });
})();
