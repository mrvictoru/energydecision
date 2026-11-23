(() => {
  "use strict";

  const status = document.getElementById("trace-status");
  const source = document.getElementById("trace-data");
  if (!status || !source) return;

  const POLICY = {dt: "DT", ppo: "PPO", replay: "Dispatch replay"};
  const METRIC = {
    dispatch: {unit: "MW", note: "Positive dispatch charges the battery; negative dispatch discharges it. Values are half-hour means."},
    soc: {unit: "MWh", note: "Stored energy is sampled at the end of each half-hour bin."},
    price: {unit: "A$/MWh", note: "Spot price is the same for all policies on each matched path. Values are half-hour means."},
    raiseBid: {unit: "%", note: "Mean raise regulation bid as a percentage of maximum power. This is the bid, not FCAS delivery."},
    lowerBid: {unit: "%", note: "Mean lower regulation bid as a percentage of maximum power. This is the bid, not FCAS delivery."},
    profit: {unit: "A$", note: "Cumulative energy and FCAS revenue less modelled wear, as in the headline profit metric."},
    energy: {unit: "A$", note: "Cumulative net energy revenue, including the cost of charging."},
    fcas: {unit: "A$", note: "Cumulative revenue across all eight modelled FCAS services."},
    wear: {unit: "A$", note: "Cumulative modelled cost of battery degradation, shown as a positive cost."},
  };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const monthFormat = new Intl.DateTimeFormat("en-AU", {month: "long", year: "numeric", timeZone: "UTC"});
  const precise = new Intl.NumberFormat("en-AU", {maximumFractionDigits: 2});
  const currency = new Intl.NumberFormat("en-AU", {style: "currency", currency: "AUD", currencyDisplay: "narrowSymbol"});
  const compact = new Intl.NumberFormat("en-AU", {notation: "compact", maximumFractionDigits: 2});

  function svgNode(tag, attrs = {}, text = "") {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text) node.textContent = text;
    return node;
  }

  function rangeFor(values) {
    let min = Math.min(0, ...values);
    let max = Math.max(0, ...values);
    if (min === max) max = min + 1;
    const power = 10 ** Math.floor(Math.log10((max - min) / 4));
    const step = [1, 2, 5, 10].find(factor => factor * power >= (max - min) / 4) * power;
    min = Math.floor(min / step) * step;
    max = Math.ceil(max / step) * step;
    return {min, max, step};
  }

  function start(data) {
    const byId = id => document.getElementById(`trace-${id}`);
    const panel = byId("viewer");
    const month = byId("month");
    const metric = byId("metric");
    const windowSelect = byId("window");
    const checks = byId("policies");
    const scrub = byId("time");
    const chart = byId("chart");
    const frame = byId("frame");
    const reading = byId("reading");
    const note = byId("note");
    const {episodes, binMinutes, capacityMwh, powerMw} = data;
    const binsPerDay = 24 * 60 / binMinutes;
    const bins = episodes[0].price.length;
    const days = bins / binsPerDay;
    let cursor = 0;
    let plot;

    episodes.forEach((episode, index) => {
      const monthName = monthFormat.format(new Date(`${episode.month}-01T00:00:00Z`));
      month.add(new Option(monthName, index));
    });
    for (let day = 0; day < days; day += 1) windowSelect.add(new Option(`Day ${day + 1} only`, day));

    function showBin(index) {
      if (!plot) return;
      cursor = Math.max(plot.start, Math.min(plot.end - 1, index));
      scrub.value = cursor - plot.start;
      const day = Math.floor(cursor / binsPerDay) + 1;
      const minutes = (cursor % binsPerDay) * binMinutes;
      const time = [Math.floor(minutes / 60), minutes % 60].map(n => String(n).padStart(2, "0")).join(":");
      const label = `Day ${day}, ${time} bin start`;
      scrub.setAttribute("aria-valuetext", label);
      plot.marker.replaceChildren(svgNode("line", {
        class: "cursor", x1: plot.x(cursor), x2: plot.x(cursor), y1: plot.top, y2: plot.bottom,
      }));
      const name = document.createElement("strong");
      name.textContent = `${label} · ${plot.unit}`;
      reading.replaceChildren(name);
      for (const line of plot.lines) {
        const value = line.values[cursor];
        plot.marker.append(svgNode("circle", {
          class: `point trace-${line.key}`, cx: plot.x(cursor), cy: plot.y(value), r: 4,
        }));
        const entry = document.createElement("span");
        const text = plot.unit === "A$" ? currency.format(value) : precise.format(value);
        entry.textContent = `${line.name}: ${text}`;
        reading.append(entry);
      }
    }

    function render() {
      const episode = episodes[Number(month.value)];
      const selected = metric.value;
      const {unit, note: explanation} = METRIC[selected];
      const spot = selected === "price";
      checks.disabled = spot;
      const lines = spot ? [{key: "price", name: "Spot price", values: episode.price}] :
        Array.from(checks.querySelectorAll("input:checked"), ({value: key}) => ({
          key, name: POLICY[key], values: episode.policies[key][selected].map(value => unit === "%" ? value * 100 : value),
        }));
      const start = windowSelect.value === "all" ? 0 : Number(windowSelect.value) * binsPerDay;
      const end = windowSelect.value === "all" ? bins : start + binsPerDay;
      const {min, max, step} = unit === "%" ? {min: 0, max: 100, step: 25} : selected === "soc" ?
        {min: 0, max: capacityMwh, step: capacityMwh / 4} : selected === "dispatch" ?
        {min: -powerMw, max: powerMw, step: powerMw / 2} :
        rangeFor(lines.flatMap(line => line.values.slice(start, end)));
      const width = Math.max(240, frame.clientWidth);
      const height = chart.clientHeight || 280;
      const left = 54;
      const right = width - 18;
      const top = 29;
      const bottom = height - 31;
      const x = index => left + (index - start + .5) / (end - start) * (right - left);
      const y = value => bottom - (value - min) / (max - min) * (bottom - top);
      chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
      chart.replaceChildren(svgNode("title", {id: "trace-chart-title"},
        `${metric.selectedOptions[0].textContent} · ${month.selectedOptions[0].textContent} · ${windowSelect.selectedOptions[0].textContent}`));
      chart.append(svgNode("text", {x: 4, y: 17}, unit));
      for (let index = 0; index <= Math.round((max - min) / step); index += 1) {
        const tick = min + step * index;
        const py = y(tick);
        chart.append(svgNode("line", {class: Math.abs(tick) < 1e-8 ? "axis" : "gridline", x1: left, x2: right, y1: py, y2: py}));
        chart.append(svgNode("text", {x: left - 6, y: py + 4, "text-anchor": "end"}, compact.format(tick + 0)));
      }
      const tickCount = width > 620 ? 6 : 3;
      for (let index = 0; index <= tickCount; index += 1) {
        const fraction = index / tickCount;
        const px = left + fraction * (right - left);
        const hours = (start + fraction * (end - start)) * binMinutes / 60;
        chart.append(svgNode("text", {x: px, y: bottom + 19, "text-anchor": index === tickCount ? "end" : "middle"}, `${hours}h`));
      }
      for (const line of lines) {
        const points = line.values.slice(start, end).map((value, i) => `${x(start + i).toFixed(2)},${y(value).toFixed(2)}`);
        chart.append(svgNode("polyline", {class: `path trace-${line.key}`, points: points.join(" ")}));
      }
      const marker = svgNode("g");
      chart.append(marker);
      plot = {x, y, top, bottom, left, right, width, start, end, lines, marker, unit};
      scrub.max = end - start - 1;
      const zoomedTotal = start > 0 && unit === "A$";
      note.textContent = lines.length ? explanation + (zoomedTotal ? " Totals still start at the beginning of the episode." : "") :
        "Select a policy to display its trace.";
      showBin(cursor);
    }

    month.addEventListener("change", render);
    metric.addEventListener("change", render);
    windowSelect.addEventListener("change", () => {
      cursor = windowSelect.value === "all" ? 0 : Number(windowSelect.value) * binsPerDay;
      render();
    });
    checks.addEventListener("change", render);
    scrub.addEventListener("input", () => showBin(plot.start + Number(scrub.value)));
    function pickBin(event) {
      if (!plot) return;
      const bounds = chart.getBoundingClientRect();
      const px = (event.clientX - bounds.left) * plot.width / bounds.width;
      const fraction = (px - plot.left) / (plot.right - plot.left);
      showBin(plot.start + Math.floor(fraction * (plot.end - plot.start)));
    }
    chart.addEventListener("click", pickBin);
    chart.addEventListener("pointermove", event => {
      if (event.pointerType === "mouse") pickBin(event);
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
    .catch(error => console.warn("The rollout viewer could not start:", error.message));
})();
