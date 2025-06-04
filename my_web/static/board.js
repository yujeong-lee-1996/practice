// $(document).ready(function () {
//     // Summernote 초기화
//     $('#content').summernote({
//         height: 300,
//         tabsize: 2,
//         callbacks: {
//             onImageUpload: function (files) {
//                 for (let i = 0; i < files.length; i++) {
//                     uploadImage(files[i]);
//                 }
//             }
//         }
//     });

//     // 글 제출 AJAX
//     $('#post-form').submit(function (e) {
//         e.preventDefault();

//         const title = $('#title').val();
//         const content = $('#content').val();  // Summernote HTML 값 읽기
//         console.log(title, content)
//         $.ajax({
//             url: '/project/board/write',  // Flask route에 맞게 경로 설정
//             type: 'POST',
//             contentType: 'application/json',
//             data: JSON.stringify({ title: title, content: content }),
//             success: function (res) {
//                 alert(res.message || "작성 완료!");
//                 $('#title').val('');
//                 $('#content').summernote('reset');  // 에디터 초기화
//             },
//             error: function (xhr) {
//                 alert('에러가 발생했습니다: ' + xhr.responseText);
//             }
//         });
//     });

//     // 이미지 업로드 함수
//     function uploadImage(file) {
//         let data = new FormData();
//         data.append("file", file);

//         $.ajax({
//             url: "/project/board/upload_image",
//             type: "POST",
//             data: data,
//             contentType: false,
//             processData: false,
//             success: function (res) {
//                 $('#content').summernote('insertImage', res.url);
//             },
//             error: function () {
//                 alert("이미지 업로드 실패");
//             }
//         });
//     }
// });


let index={
    init:function(){
        $("#btn-save").on("click",()=>{
            this.save();
        });
        $("#btn-delete").on("click",()=>{
            this.deleteById();
        });
        $("#btn-update").on("click",()=>{
            this.update();
        });
        $("#btn-reply-save").on("click",()=>{
            this.replySave();
        });

    },

    save:function (){
        //  alert("user의 save함수됨");
        let data={
            title:$("#title").val(),
            content:$("#content").val()
        };
        $.ajax({
            type:"POST",
            url:"/project/board/write",
            data: JSON.stringify(data), //json 문자열로 변환 http body데이터
            contentType:"application/json; charset=utf-8",//body데이터가 어떤 타입인지(MIME)
            dataType:"json" //요청을 서버로해서 응답이 왔을 때 기본적으로 모든것이 String (생긴게 json이라면)=>javascript object로 변환
        }).done(function (resp){
            alert("글쓰기가 완료되었습니다.");
            $("#title").val("");
            $("#content").summernote('reset');
            $("#content").val("");
        }).fail(function (error){
            alert(JSON.stringify(error));
        });
    },

    deleteById:function (){
        let id=$("#id").text();

        $.ajax({
            type:"DELETE",
            url:"/api/board/"+id,
            dataType:"json" //요청을 서버로해서 응답이 왔을 때 기본적으로 모든것이 String (생긴게 json이라면)=>javascript object로 변환
        }).done(function (resp){
            alert("삭제가 완료되었습니다.");
            location.href="/";
        }).fail(function (error){
            alert(JSON.stringify(error));
        });
    },

    update:function (){
        let id=$("#id").val();

        let data={
            title:$("#title").val(),
            content:$("#content").val()
        };

        $.ajax({
            type:"PUT",
            url:"/api/board/"+id,
            data: JSON.stringify(data), //json 문자열로 변환 http body데이터
            contentType:"application/json; charset=utf-8",//body데이터가 어떤 타입인지(MIME)
            dataType:"json" //요청을 서버로해서 응답이 왔을 때 기본적으로 모든것이 String (생긴게 json이라면)=>javascript object로 변환
        }).done(function (resp){
            alert("글수정이 완료되었습니다.");
            location.href="/";
        }).fail(function (error){
            alert(JSON.stringify(error));
        });
    },

    replySave: function (){
        let data={
            userId: $("#userId").val(),
            boardId: $("#boardId").val(),
            content: $("#reply-content").val()
        };

        console.log(data);
        $.ajax({
            type:"POST",
            url:`/api/board/${data.boardId}/reply`,
            data: JSON.stringify(data), //json 문자열로 변환 http body데이터
            contentType:"application/json; charset=utf-8",//body데이터가 어떤 타입인지(MIME)
            dataType:"json" //요청을 서버로해서 응답이 왔을 때 기본적으로 모든것이 String (생긴게 json이라면)=>javascript object로 변환
        }).done(function (resp){
            alert("댓글작성이 완료되었습니다.");
            location.href=`/board/${data.boardId}`;
        }).fail(function (error){
            alert(JSON.stringify(error));
        });
    },

    replyDelete: function (boardId, replyId){
        $.ajax({
            type:"DELETE",
            url:`/api/board/${boardId}/reply/${replyId}`,
            dataType:"json" //요청을 서버로해서 응답이 왔을 때 기본적으로 모든것이 String (생긴게 json이라면)=>javascript object로 변환
        }).done(function (resp){
            alert("댓글이 삭제되었습니다.");
            location.href=`/board/${boardId}`;
        }).fail(function (error){
            alert(JSON.stringify(error));
        });
    },


}

index.init();