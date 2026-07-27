"use strict";

// Page 2 — Select Album: pick a previously processed album from the
// polaroid gallery, or click the leading "+" card to add a new one
// (navigates to the Add Album page).

// Hover slideshow tick rate for polaroid preview cycling, in ms.
const SLIDESHOW_INTERVAL_MS = 700;

// ------------------------------------------------------------------ //
// Previously processed albums — polaroid gallery
// ------------------------------------------------------------------ //
function albumThumbUrl(album, file) {
  return `/api/albums/thumb?id=${encodeURIComponent(album.name)}&file=${encodeURIComponent(file)}`;
}

function buildAddAlbumCard() {
  const card = $("<div>", { class: "polaroid polaroid-add" });
  const photo = $("<div>", { class: "polaroid-photo polaroid-add-photo" });
  photo.append($("<span>", { class: "polaroid-add-plus" }).text("+"));
  const caption = $("<div>", { class: "polaroid-caption" });
  caption.append($("<div>", { class: "polaroid-name" }).text("New Album"));

  card.append(photo, caption);
  card.on("click", () => { window.location.href = "/add-album"; });
  return card;
}


function buildPolaroidCard(album) {
  const urls = (album.previewImages || []).map((f) => albumThumbUrl(album, f));
  const card = $("<div>", { class: "polaroid" });
  const photo = $("<div>", { class: "polaroid-photo" });

  if (urls.length) {
    const startUrl = urls[Math.floor(Math.random() * urls.length)];
    const img = $("<img>", { src: startUrl, alt: album.displayName || album.name });
    img.data("original", startUrl);
    img.data("urls", urls);
    photo.append(img);
  } else {
    photo.append($("<div>", { class: "polaroid-empty muted" }).text("No preview"));
  }

  const deleteBtn = $("<button>", {
    type: "button",
    class: "polaroid-delete",
    title: "Delete album",
    html: "&times;",
  });
  photo.append(deleteBtn);

  const caption = $("<div>", { class: "polaroid-caption" });
  caption.append($("<div>", { class: "polaroid-name" }).text(album.displayName || album.name));
  caption.append($("<div>", { class: "polaroid-date" }).text(album.createdDisplay || ""));

  card.append(photo, caption);

  // Hover slideshow: cycle the thumbnail through a random sample of the
  // album's sharp images while hovered; reset to the original still on
  // mouse-out. (The delete button's fade in/out is pure CSS — see
  // .polaroid:hover .polaroid-delete in style.css.)
  let slideTimer = null;
  card.on("mouseenter", () => {
    const img = card.find("img");
    const imgUrls = img.data("urls");
    if (!imgUrls || imgUrls.length < 2) return;
    slideTimer = setInterval(() => {
      img.attr("src", imgUrls[Math.floor(Math.random() * imgUrls.length)]);
    }, SLIDESHOW_INTERVAL_MS);
  });
  card.on("mouseleave", () => {
    if (slideTimer) { clearInterval(slideTimer); slideTimer = null; }
    const img = card.find("img");
    const original = img.data("original");
    if (original) img.attr("src", original);
  });

  deleteBtn.on("click", async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete album "${album.displayName || album.name}"? This permanently removes its output folder and cannot be undone.`)) {
      return;
    }
    deleteBtn.prop("disabled", true);
    try {
      await apiPost("/api/albums/delete", { id: album.name });
      if (slideTimer) clearInterval(slideTimer);
      card.fadeOut(180, function () { $(this).remove(); });
    } catch (err) {
      alert("Failed to delete: " + err.message);
      deleteBtn.prop("disabled", false);
    }
  });

  card.on("click", async () => {
    try {
      await apiPost("/api/albums/select", { id: album.name });
      window.location.href = "/review";
    } catch (err) {
      alert("Failed to resume: " + err.message);
    }
  });

  return card;
}

async function loadAlbums() {
  const data = await apiGet("/api/albums");
  const gallery = $("#albumGallery").empty();
  gallery.append(buildAddAlbumCard());
  data.albums.forEach((album) => gallery.append(buildPolaroidCard(album)));
}

$(function () {
  loadAlbums().catch(() => {});
});

