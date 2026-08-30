"use strict";

// Shared "viewport" zoom/pan math + a reusable "frame" (modal image window)
// built on top of it. Used by the review page (viewport embedded directly
// in the page, no frame) and the cluster page (viewport hosted inside a
// modal frame). See static/js/review.js and static/js/cluster.js.
const Viewport = (function () {
  // Given an image's natural size and the available space to display it
  // in, compute the classic object-fit: contain box (the "zoom-to-fit"
  // size), in pixels.
  function computeContainFit(naturalW, naturalH, availW, availH) {
    if (!naturalW || !naturalH || !availW || !availH) return null;
    const scale = Math.min(availW / naturalW, availH / naturalH);
    return { width: Math.round(naturalW * scale), height: Math.round(naturalH * scale) };
  }

  // A zoom/pan controller: holds the current zoom level + pan offset and
  // applies them (via a CSS transform) to whatever set of <img> elements is
  // passed in at call time. Callers own which images are "live" (e.g. the
  // review page re-queries its currently displayed preview cells so up to 4
  // images can share one zoom/pan state), so the same controller instance
  // can be reused across image/group navigation (and, on the review page,
  // across the annotated/original toggle — see toggleViewMode/renderMain in
  // review.js) without resetting.
  //
  // Pan is tracked as a `translate(tx, ty) scale(zoom)` transform (in that
  // order) with transform-origin pinned to the image's top-left corner, so
  // tx/ty are plain screen pixels regardless of zoom: dragging the mouse by
  // (dx, dy) always just adds (dx, dy) to (tx, ty), and "zoom centered on a
  // point" is solved by picking a new tx/ty that keeps that point's screen
  // position unchanged (see zoomTo below). Since the <img> itself is always
  // sized to exactly fill its (fixed-size, non-scaling) viewport parent at
  // zoom 1, that parent's rect doubles as the img's un-transformed base
  // rect for both of those calculations.
  function createZoomController({ min = 1, max = 6, step = 0.25 } = {}) {
    let zoom = min;
    let tx = 0;
    let ty = 0;

    function baseRectOf(imgEl) {
      return imgEl.parentElement.getBoundingClientRect();
    }

    // Keeps the pan offset within the range that leaves the (scaled) image
    // fully covering its viewport — i.e. no empty gap at any edge.
    function clamp(imgEl) {
      const r = baseRectOf(imgEl);
      tx = Math.min(0, Math.max(r.width * (1 - zoom), tx));
      ty = Math.min(0, Math.max(r.height * (1 - zoom), ty));
    }

    function apply($imgs) {
      if (zoom <= min) {
        tx = 0;
        ty = 0;
        $imgs.removeClass("zoomed").css({ transform: "", "transform-origin": "" });
        return;
      }
      $imgs.addClass("zoomed").css({
        transform: `translate(${tx}px, ${ty}px) scale(${zoom})`,
        "transform-origin": "0 0",
      });
    }

    function resetToFit($imgs) {
      zoom = min;
      apply($imgs);
    }

    // Changes zoom to `newZoom`, keeping the point under (clientX, clientY)
    // — relative to `imgEl` — fixed on screen, if given; otherwise zooms
    // centered on the image itself.
    function zoomTo($imgs, imgEl, newZoom, clientX, clientY) {
      newZoom = Math.min(max, Math.max(min, newZoom));
      if (imgEl) {
        const r = baseRectOf(imgEl);
        const cx = clientX != null ? clientX : r.left + r.width / 2;
        const cy = clientY != null ? clientY : r.top + r.height / 2;
        const localX = (cx - r.left - tx) / zoom;
        const localY = (cy - r.top - ty) / zoom;
        tx += localX * (zoom - newZoom);
        ty += localY * (zoom - newZoom);
      }
      zoom = newZoom;
      if (imgEl) clamp(imgEl);
      apply($imgs);
    }

    // Zoom to 100% (actual pixel size) of `imgEl`, anchored on (clientX,
    // clientY) if given, otherwise centered on the image.
    function zoomToActual($imgs, imgEl, clientX, clientY) {
      if (!imgEl || !imgEl.naturalWidth) return;
      const rect = baseRectOf(imgEl);
      const nativeZoom = imgEl.naturalWidth / rect.width;
      zoomTo($imgs, imgEl, nativeZoom, clientX, clientY);
    }

    function adjustByStep($imgs, delta, imgEl, clientX, clientY) {
      zoomTo($imgs, imgEl, zoom + delta, clientX, clientY);
    }

    // Click-to-toggle: zooms to actual size if currently at fit, or back to
    // fit if already zoomed in.
    function toggleClick($imgs, imgEl, clientX, clientY) {
      if (zoom > min) {
        resetToFit($imgs);
        return;
      }
      zoomToActual($imgs, imgEl, clientX, clientY);
    }

    // Drag-to-pan. Starts tracking a drag anchored at (clientX, clientY);
    // returns false (and starts nothing) while at "fit" zoom, since there's
    // no room to pan there. `imgEl` is remembered only to know which
    // viewport to clamp against as the drag proceeds.
    let dragImg = null;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragStartTx = 0;
    let dragStartTy = 0;

    function dragStart(imgEl, clientX, clientY) {
      if (zoom <= min) return false;
      dragImg = imgEl;
      dragStartX = clientX;
      dragStartY = clientY;
      dragStartTx = tx;
      dragStartTy = ty;
      return true;
    }

    function dragMove($imgs, clientX, clientY) {
      if (!dragImg) return;
      tx = dragStartTx + (clientX - dragStartX);
      ty = dragStartTy + (clientY - dragStartY);
      clamp(dragImg);
      apply($imgs);
    }

    function dragEnd() {
      dragImg = null;
    }

    function isZoomed() {
      return zoom > min;
    }

    return {
      apply, resetToFit, zoomToActual, adjustByStep, toggleClick, isZoomed,
      dragStart, dragMove, dragEnd,
      get min() { return min; },
      get max() { return max; },
      get step() { return step; },
    };
  }

  // A self-contained modal "frame" hosting a single-image viewport: darkens
  // the background, locks page scroll, and shows a close [x] button. Zoom
  // interactions over the image match the shared viewport behavior (scroll
  // to zoom, drag to pan while zoomed, click to toggle fit/actual, '1' to
  // zoom to actual). Clicking anywhere outside the image, the close button,
  // or any other key (including Esc) closes the window immediately.
  //
  // `opts.originalPath` / `opts.aiEditKey` (both optional, cluster.js is
  // the only caller that passes them) add a toolbar below the image: a
  // readonly path box plus an "AI edit" button (see static/js/ai-edit.js).
  // Clicks/keystrokes inside that toolbar are excluded from the "anything
  // closes this window" behavior below, so the path can actually be
  // selected/copied and the button clicked without the window vanishing
  // first.
  function showImageWindow(imageUrl, opts) {
    opts = opts || {};
    const zoomCtl = createZoomController({ min: 1, max: 6, step: 0.25 });

    const $viewport = $("<div>", { class: "image-window-viewport" });
    const $img = $("<img>", { src: imageUrl, alt: "" });
    $viewport.append($img);
    const $closeBtn = $("<button>", { type: "button", class: "image-window-close", "aria-label": "Close" }).html("&times;");
    const $backdrop = $("<div>", { class: "image-window-backdrop" }).append($viewport, $closeBtn);
    if (opts.originalPath) {
      const $toolbar = $("<div>", { class: "image-window-toolbar" });
      $toolbar.append(
        $("<input>", { type: "text", class: "image-window-path", readonly: true }).val(opts.originalPath)
      );
      if (opts.aiEditKey) {
        $toolbar.append(
          $("<button>", { type: "button", class: "btn btn-sm" }).text("AI edit")
            .on("click", () => AiEdit.run(opts.aiEditKey))
        );
      }
      $backdrop.append($toolbar);
    }
    $("body").append($backdrop).addClass("image-window-open");

    function imgs() {
      return $viewport.find("img");
    }

    function lock() {
      const box = computeContainFit($img[0].naturalWidth, $img[0].naturalHeight, window.innerWidth * 0.9, window.innerHeight * 0.9);
      if (box) $viewport.css({ width: box.width + "px", height: box.height + "px" });
    }
    if ($img[0].complete) lock();
    else $img.on("load", lock);
    $(window).on("resize.imageWindow", lock);

    // Drag-to-pan while zoomed in. `dragging` tracks whether a drag started
    // on the image is still in progress; `dragMoved` distinguishes an actual
    // drag from a plain click so the click handler below (fit/actual toggle)
    // doesn't also fire once the drag ends.
    let dragging = false;
    let dragMoved = false;
    $viewport.on("mousedown", "img", function (e) {
      e.preventDefault(); // suppress the browser's native image-ghost drag
      dragMoved = false;
      dragging = zoomCtl.dragStart(this, e.clientX, e.clientY);
    });
    $(document).on("mousemove.imageWindowDrag", function (e) {
      if (!dragging) return;
      dragMoved = true;
      zoomCtl.dragMove(imgs(), e.clientX, e.clientY);
    });
    $(document).on("mouseup.imageWindowDrag", function () {
      dragging = false;
    });
    $viewport.on("wheel", "img", function (e) {
      e.preventDefault();
      const oe = e.originalEvent;
      const delta = oe.deltaY < 0 ? zoomCtl.step : -zoomCtl.step;
      zoomCtl.adjustByStep(imgs(), delta, this, oe.clientX, oe.clientY);
    });
    $viewport.on("click", "img", function (e) {
      if (dragMoved) {
        dragMoved = false;
        return;
      }
      zoomCtl.toggleClick(imgs(), this, e.clientX, e.clientY);
    });

    // Clicking outside the viewport (the backdrop) or the close button closes.
    // Clicks inside the toolbar (path box / AI edit button) are excluded too.
    $backdrop.on("click", function (e) {
      if ($(e.target).closest(".image-window-viewport, .image-window-toolbar").length) return;
      close();
    });
    $closeBtn.on("click", close);

    // Any keydown closes the window, except '1' which zooms to actual size
    // (matching the shared viewport hotkey) — captured ahead of any other
    // page-level keydown handler (e.g. the cluster page's Delete-key
    // handler) so it never also fires while the window is open.
    function onKeydown(e) {
      // Let normal typing/selection (e.g. Ctrl+A/Ctrl+C on the readonly path
      // box) work while focus is in the toolbar, instead of closing the
      // window on the very first keystroke.
      if ($(e.target).closest(".image-window-toolbar").length) return;
      e.stopPropagation();
      if (e.key === "1") {
        zoomCtl.zoomToActual(imgs(), $img[0]);
        return;
      }
      e.preventDefault();
      close();
    }
    document.addEventListener("keydown", onKeydown, true);

    function close() {
      document.removeEventListener("keydown", onKeydown, true);
      $(document).off("mousemove.imageWindowDrag mouseup.imageWindowDrag");
      $(window).off("resize.imageWindow");
      $("body").removeClass("image-window-open");
      $backdrop.remove();
    }

    return { close };
  }

  return { computeContainFit, createZoomController, showImageWindow };
})();
