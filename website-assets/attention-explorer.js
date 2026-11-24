(() => {
  "use strict";

  const groups = document.getElementById("attention-groups");
  const queryControl = document.getElementById("attention-query-control");
  const query = document.getElementById("attention-query");
  const tokens = [...document.querySelectorAll("#attention-tokens li")];
  const note = document.getElementById("attention-mask-note");
  const stateNote = note.textContent;
  const first = document.getElementById("head-first");
  const second = document.getElementById("head-second");
  const key = document.getElementById("head-key");
  const value = document.getElementById("head-value");

  groups.addEventListener("change", event => {
    const group = Number(event.target.value);
    first.textContent = `Query ${2 * group - 1}`;
    second.textContent = `Query ${2 * group}`;
    key.textContent = `K${group}`;
    value.textContent = `V${group}`;
  });

  query.addEventListener("change", () => {
    const index = Number(query.value);
    tokens.forEach((token, i) => {
      token.classList.toggle("is-query", i === index);
      token.classList.toggle("is-masked", i > index);
      token.querySelector("small").textContent = i === index ? "Query" : i > index ? "Masked" : "Visible";
    });
    note.textContent = index === 4 ? stateNote : index === tokens.length - 1 ?
      "All shown tokens are visible here. Tokens after this action are still masked." :
      "This query can use its own token and tokens to its left in the sequence, but none to its right.";
  });

  groups.hidden = false;
  queryControl.hidden = false;
})();
