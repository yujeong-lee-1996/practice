$(document).ready(function () {
  $('#summernote').summernote({
    height: 300,
    tabsize: 2,
    maxHeight: 300,
    placeholder: '내용을 작성하세요.',
    callbacks: {
      onImageUpload: function (files) {
        const formData = new FormData();
        formData.append('file', files[0]);

        $.ajax({
          url: '/project/board/upload_image',
          method: 'POST',
          data: formData,
          processData: false,
          contentType: false,
          success: function (response) {
            $('#summernote').summernote('insertImage', response.url);
          },
          error: function () {
            alert('이미지 업로드 실패');
          }
        });
      }
    }
  });

  // 글 작성
  $('#post-form').on('submit', function (e) {
    e.preventDefault();

    const title = $('#title').val().trim();
    const content = $('#summernote').summernote('code').trim();

    if (!title || !content) {
      alert('제목과 내용을 모두 입력해주세요.');
      return;
    }

    $.ajax({
      type: 'POST',
      url: '/project/board/write',
      contentType: 'application/json',
      data: JSON.stringify({ title, content }),
      success: function (response) {
        alert(response.message);
        window.location.href = '/project/board';
      },
      error: function () {
        alert('글 작성 중 오류가 발생했습니다.');
      }
    });
  });

  // 글 수정
  $('#edit-form').on('submit', function (e) {
    e.preventDefault();

    const title = $('#title').val().trim();
    const content = $('#summernote').summernote('code').trim();

    if (!title || !content) {
      alert('제목과 내용을 모두 입력해주세요.');
      return;
    }

    $.ajax({
      type: 'POST',
      url: `/project/board/post/${POST_ID}/edit`,
      contentType: 'application/json',
      data: JSON.stringify({ title, content }),
      success: function (response) {
        alert(response.message);
        window.location.href = `/project/board/post/${POST_ID}`;
      },
      error: function () {
        alert('글 수정 중 오류가 발생했습니다.');
      }
    });
  });
});
