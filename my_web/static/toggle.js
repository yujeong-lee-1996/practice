document.addEventListener("DOMContentLoaded", function () {
    const toggleBtn = document.getElementById("toggleSidebarBtn");
    const sidebar = document.getElementById("sidebar");
  
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("closed");
    });
  });
  