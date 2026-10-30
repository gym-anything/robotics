(function attachTeleopControlRuntime(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.TeleopControlRuntime = api;
}(typeof globalThis === "object" ? globalThis : this, () => {
  "use strict";

  function normalizedIntent(intent) {
    if (!intent || !["left", "right"].includes(intent.arm)) {
      throw new TypeError("control intent requires a left or right arm");
    }
    if (!Array.isArray(intent.keys)) {
      throw new TypeError("control intent keys must be an array");
    }
    return Object.freeze({
      arm: intent.arm,
      keys: Object.freeze([...new Set(intent.keys.map(String))].sort()),
      released: intent.released === true,
      generation: Number(intent.generation),
    });
  }

  function sameIntent(left, right) {
    return Boolean(
      left
      && right
      && left.arm === right.arm
      && left.released === right.released
      && left.generation === right.generation
      && left.keys.length === right.keys.length
      && left.keys.every((key, index) => key === right.keys[index])
    );
  }

  class InputIntentQueue {
    constructor() {
      this._items = [];
    }

    get length() {
      return this._items.length;
    }

    push(intent) {
      const item = normalizedIntent(intent);
      if (sameIntent(this._items[this._items.length - 1], item)) return false;
      this._items.push(item);
      return true;
    }

    shift(generation) {
      while (this._items.length) {
        const item = this._items.shift();
        if (item.generation === generation) return item;
      }
      return null;
    }

    clear() {
      this._items.length = 0;
    }

    snapshot() {
      return this._items.map((item) => ({
        arm: item.arm,
        keys: [...item.keys],
        released: item.released,
        generation: item.generation,
      }));
    }
  }

  class RefreshLatch {
    constructor() {
      this.pending = false;
      this.running = false;
    }

    request() {
      this.pending = true;
    }

    begin() {
      if (this.running || !this.pending) return false;
      this.pending = false;
      this.running = true;
      return true;
    }

    finish() {
      if (!this.running) throw new Error("refresh is not running");
      this.running = false;
      return this.pending;
    }
  }

  function isEditableElement(target) {
    for (let element = target; element; element = element.parentElement) {
      const tagName = String(element.tagName || "").toUpperCase();
      if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") {
        return true;
      }
      if (element.isContentEditable === true) return true;
      if (typeof element.getAttribute === "function") {
        const contentEditable = element.getAttribute("contenteditable");
        if (contentEditable === "" || contentEditable === "true" || contentEditable === "plaintext-only") {
          return true;
        }
      }
    }
    return false;
  }

  return Object.freeze({ InputIntentQueue, RefreshLatch, isEditableElement, sameIntent });
}));
