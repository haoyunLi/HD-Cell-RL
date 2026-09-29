const query = new URLSearchParams(window.location.search);
const requestedSlide = Number.parseInt(query.get("slide"), 10);
const requestedFragment = Number.parseInt(query.get("fragment"), 10);

if (Number.isInteger(requestedSlide) && requestedSlide >= 0) {
  const fragment = Number.isInteger(requestedFragment) && requestedFragment >= -1
    ? requestedFragment
    : -1;
  window.location.hash = fragment < 0 ? `/${requestedSlide}` : `/${requestedSlide}/0/${fragment}`;
}

Reveal.initialize({
  hash: true,
  history: true,
  controls: true,
  controlsTutorial: false,
  progress: true,
  center: false,
  width: 1600,
  height: 900,
  margin: 0,
  minScale: 0.2,
  maxScale: 1.5,
  transition: "fade",
  transitionSpeed: "fast",
  backgroundTransition: "fade",
  slideNumber: "c/t",
  viewDistance: 3,
  scrollActivationWidth: 0,
  pdfSeparateFragments: false,
});
