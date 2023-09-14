function sendJsonData(url, data, hash) {
    var json_asocc = {
      'data':data,
      'hash':hash
    };
     
    var json_text = JSON.stringify(json_asocc);
 
    xhr = new XMLHttpRequest;
    xhr.onload = function() {
        var res = xhr.responseText;
        if (res.length>0) alert(res);
    };
    xhr.onerror = function() {
        alert("error!");
    }
    xhr.open('post', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(json_text);
}
