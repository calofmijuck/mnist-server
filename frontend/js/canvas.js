const lineWidth = 15;

let canvas, ctx;

// Track mouse position and left-button status 
let mouseX, mouseY, mouseDown = false;

// Track touch position
let touchX, touchY;

// Draw a dot at a specific position on the supplied canvas
// Param: A canvas context, the x position, the y position, the size of the dot
const drawDot = (ctx, xPos, yPos, dotSize) => {
    ctx.fillStyle = "black";

    ctx.beginPath();
    ctx.arc(xPos, yPos, dotSize, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
};

const clearCanvas = () => {
    ctx.beginPath();
    ctx.rect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fill();
};

const onMouseDown = () => {
    mouseDown = true;
    drawDot(ctx, mouseX, mouseY, lineWidth);
};

const onMouseUp = () => {
    mouseDown = false;
};

const onMouseMove = (event) => {
    getMousePosition(event);

    if (mouseDown) {
        drawDot(ctx, mouseX, mouseY, lineWidth);
    }
};

const getMousePosition = (event) => {
    if (event.offsetX) {
        mouseX = event.offsetX;
        mouseY = event.offsetY;
    } else if (event.layerX) {
        mouseX = event.layerX;
        mouseY = event.layerY;
    }
};

const onTouchStart = () => {
    getTouchPosition();

    drawDot(ctx, touchX, touchY, lineWidth);
    // event.preventDefault();
};

const onTouchMove = (event) => {
    getTouchPosition(event);

    drawDot(ctx, touchX, touchY, lineWidth);
    // event.preventDefault();
};

/* 
    Get the touch position relative to the top-left of the canvas
    When we get the raw values of pageX and pageY below, they take into account the scrolling on the page
    but not the position relative to our target div. We will adjust them using "target.offsetLeft" and
    "target.offsetTop" to get the correct values in relation to the top left of the canvas.
*/
const getTouchPosition = (event) => {
    if (!event.touches) {
        return;
    }

    if (event.touches.length === 1) {
        let touch = event.touches[0];
        touchX = touch.pageX - touch.target.offsetLeft;
        touchY = touch.pageY - touch.target.offsetTop;
    }
};

const init = () => {
    canvas = document.getElementById("sketchpad");

    if (canvas.getContext) {
        ctx = canvas.getContext("2d");
    }

    if (!ctx) {
        return
    }

    canvas.addEventListener("mousedown", onMouseDown, false);
    canvas.addEventListener("mousemove", onMouseMove, false);
    window.addEventListener("mouseup", onMouseUp, false);

    canvas.addEventListener("touchstart", onTouchStart, false);
    canvas.addEventListener("touchmove", onTouchMove, false);

    clearCanvas();
};
