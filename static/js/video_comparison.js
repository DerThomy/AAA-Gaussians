// Written by Lukas Radl, April 2024
// Modified by Thomas Köhler, August 2025
// Adapted from the following sources
// Ref-NeRF     https://dorverbin.github.io/refnerf/
// Reconfusion  https://reconfusion.github.io/
// DICS         https://github.com/abelcabezaroman/definitive-image-comparison-slider
var strokeColor = "#FFFFFFDD";

var vidShow = 0;

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var vidWidth = vid.videoWidth/2;

    var position = 0.25

    var subVidHeight = vid.videoHeight;
    var interm_pos = 0;

    var mergeContext = videoMerge.getContext("2d");
    
    if (vid.readyState > 3) {
        vid.muted = true;
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - (bcr.x + window.scrollX)) / bcr.width);
        }

        videoMerge.addEventListener("mousemove",  trackLocation, false); 
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove",  trackLocationTouch, false);


        function drawLoop() {
            mergeContext.drawImage(vid, 0, vidShow * subVidHeight, vidWidth, subVidHeight, 0, 0, vidWidth, subVidHeight);
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart+vidWidth, vidShow * subVidHeight, colWidth, subVidHeight, colStart, 0, colWidth, subVidHeight);
            requestAnimationFrame(drawLoop);

            var currX = vidWidth * position;
            
            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth*position, 0);
            mergeContext.lineTo(vidWidth*position, subVidHeight);
            mergeContext.closePath()
            mergeContext.strokeStyle = strokeColor;
            mergeContext.lineWidth = 2;            
            mergeContext.stroke();

            var arrowPosY2 = subVidHeight / 2;
            var arrowW = subVidHeight / 70;
            var arrowL = subVidHeight / 150;
            var arrowoffsetL = subVidHeight / 150;

            // draw (similar to dics)
            mergeContext.beginPath();
            mergeContext.moveTo(currX + arrowL + arrowoffsetL, arrowPosY2 - arrowW/2);
            mergeContext.lineTo(currX + 2*arrowL + arrowoffsetL, arrowPosY2 );
            mergeContext.lineTo(currX + arrowL + arrowoffsetL, arrowPosY2 + arrowW/2);

            mergeContext.strokeStyle = strokeColor;
            mergeContext.stroke();

            // draw (similar to dics)
            mergeContext.beginPath();
            mergeContext.moveTo(currX - arrowL - arrowoffsetL, arrowPosY2 - arrowW/2);
            mergeContext.lineTo(currX - 2*arrowL - arrowoffsetL, arrowPosY2 );
            mergeContext.lineTo(currX - arrowL - arrowoffsetL, arrowPosY2 + arrowW/2);

            mergeContext.strokeStyle = strokeColor;
            mergeContext.stroke();
            
        }
        requestAnimationFrame(drawLoop);
    } 
}

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};
    
function resizeAndPlay(element)
{
  var cv = document.getElementById(element.id + "Merge");
  cv.width = element.videoWidth/2;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "1px";
  element.style.position = "absolute";
    
  playVids(element.id);
}
