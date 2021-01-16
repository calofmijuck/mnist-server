const ENDPOINT = "http://www.zxcvber.com:8888";

const predict = () => {
    let dataURL = canvas.toDataURL();

    $.ajax({
        type: "POST",
        url: ENDPOINT,
        data: {
            data: dataURL
        },
        contentType: "application/json",
        dataType: "json"
    }).done((response) => {
        $("#answer").text(response.ans);
    });
};
