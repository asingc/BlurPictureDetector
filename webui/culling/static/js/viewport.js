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

  // A zoom/pan controller: holds the current zoom level and applies it (via
  // a CSS transform) to whatever set of <img> elements is passed in at call
  // time. Callers own which images are "live" (e.g. the review page
  // re-queries its currently displayed preview cells so up to 4 images can
  // share one zoom level), so the same controller instance can be reused
  // across image/group navigation without resetting.
  function createZoomController({ min = 1, max = 6, step = 0.25 } = {}) {
    let zoom = min;

    function origin(clientX, clientY, imgEl) {
      const rect = imgEl.getBoundingClientRect();
      return { x: ((clientX - rect.left) / rect.width) * 100, y: ((clientY - rect.top) / rect.height) * 100 };
    }

    function apply($imgs, originX, originY) {
      if (zoom <= min) {
        $imgs.removeClass("zoomed").css({ transform: "", "transform-origin": "" });
        return;
      }
      const o = originX != null && originY != null ? `${originX}% ${originY}%` : "50% 50%";
      $imgs.addClass("zoomed").css({ transform: `scale(${zoom})`, "transform-origin": o });
    }

    function resetToFit($imgs) {
      zoom = min;
      apply($imgs);
    }

    // Zoom to 100% (actual pixel size) of `imgEl`, centered on (clientX,
    // clientY) if given, otherwise centered on the image.
    function zoomToActual($imgs, imgEl, clientX, clientY) {
      if (!imgEl || !imgEl.naturalWidth) return;
      const rect = imgEl.getBoundingClientRect();
      const nativeZoom = imgEl.naturalWidth / rect.width;
      zoom = Math.min(max, Math.max(min, nativeZoom));
      if (clientX != null && clientY != null) {
        const o = origin(clientX, clientY, imgEl);
        apply($imgs, o.x, o.y);
      } else {
        apply($imgs);
      }
    }

    function adjustByStep($imgs, delta, imgEl, clientX, clientY) {
      zoom = Math.min(max, Math.max(min, zoom + delta));
      if (imgEl && clientX != null) {
        const o = origin(clientX, clientY, imgEl);
        apply($imgs, o.x, o.y);
      } else {
        apply($imgs);
      }
    }

    // Pans a zoomed-in image proportionally to the cursor position. No-op
    // while at "fit" zoom.
    function panTo($imgs, imgEl, clientX, clientY) {
      if (zoom <= min) return;
      const o = origin(clientX, clientY, imgEl);
      apply($imgs, o.x, o.y);
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

    function isZoomed() {
      return zoom > min;
    }

    return {
      apply, origin, resetToFit, zoomToActual, adjustByStep, panTo, toggleClick, isZoomed,
      get min() { return min; },
      get max() { return max; },
      get step() { return step; },
    };
  }

  // A self-contained modal "frame" hosting a single-image viewport: darkens
  // the background, locks page scroll, and shows a close [x] button. Zoom
  // interactions over the image match the shared viewport behavior (scroll
  // to zoom, hover to pan while zoomed, click to toggle fit/actual, '1' to
  // zoom to actual). Clicking anywhere outside the image, the close button,
  // or any other key (including Esc) closes the window immediately.
  function showImageWindow(imageUrl) {
    const zoomCtl = createZoomController({ min: 1, max: 6, step: 0.25 });

    const $viewport = $("<div>", { class: "image-window-viewport" });
    const $img = $("<img>", { src: imageUrl, alt: "" });
    $viewport.append($img);
    const $closeBtn = $("<button>", { type: "button", class: "image-window-close", "aria-label": "Close" }).html("&times;");
    const $backdrop = $("<div>", { class: "image-window-backdrop" }).append($viewport, $closeBtn);
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

    $viewport.on("mousemove", "img", function (e) {
      zoomCtl.panTo(imgs(), this, e.clientX, e.clientY);
    });
    $viewport.on("wheel", "img", function (e) {
      e.preventDefault();
      const oe = e.originalEvent;
      const delta = oe.deltaY < 0 ? zoomCtl.step : -zoomCtl.step;
      zoomCtl.adjustByStep(imgs(), delta, this, oe.clientX, oe.clientY);
    });
    $viewport.on("click", "img", function (e) {
      zoomCtl.toggleClick(imgs(), this, e.clientX, e.clientY);
    });

    // Clicking outside the viewport (the backdrop) or the close button closes.
    $backdrop.on("click", function (e) {
      if ($(e.target).closest(".image-window-viewport").length) return;
      close();
    });
    $closeBtn.on("click", close);

    // Any keydown closes the window, except '1' which zooms to actual size
    // (matching the shared viewport hotkey) — captured ahead of any other
    // page-level keydown handler (e.g. the cluster page's Delete-key
    // handler) so it never also fires while the window is open.
    function onKeydown(e) {
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
      $(window).off("resize.imageWindow");
      $("body").removeClass("image-window-open");
      $backdrop.remove();
    }

    return { close };
  }

  return { computeContainFit, createZoomController, showImageWindow };
})();
