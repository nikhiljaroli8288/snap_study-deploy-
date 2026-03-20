// Dark Mode Toggle Logic
document.addEventListener('DOMContentLoaded', () => {
    const htmlElement = document.documentElement;

    function isDark() {
        return htmlElement.classList.contains('dark');
    }

    // Update ALL toggle icons/labels on the page
    function updateAllToggles() {
        const dark = isDark();
        // Main sidebar/header toggle
        const icon = document.getElementById('theme-toggle-icon');
        if (icon) icon.textContent = dark ? 'light_mode' : 'dark_mode';

        // Settings page toggle
        const settingsIcon = document.getElementById('settings-toggle-icon');
        const settingsLabel = document.getElementById('settings-toggle-label');
        if (settingsIcon) settingsIcon.textContent = dark ? 'light_mode' : 'dark_mode';
        if (settingsLabel) settingsLabel.textContent = dark ? 'Light Mode' : 'Dark Mode';
    }

    // Apply theme
    function setTheme(theme) {
        if (theme === 'dark') {
            htmlElement.classList.add('dark');
        } else {
            htmlElement.classList.remove('dark');
        }
        localStorage.setItem('theme', theme);
        updateAllToggles();
    }

    function toggleTheme() {
        setTheme(isDark() ? 'light' : 'dark');
    }

    // Initialize icons on page load
    updateAllToggles();

    // Bind main toggle button (sidebar or header)
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // Bind settings page toggle
    const settingsToggleBtn = document.getElementById('settings-theme-toggle');
    if (settingsToggleBtn) {
        settingsToggleBtn.addEventListener('click', toggleTheme);
    }

    // Toggle Study Depth Buttons
    const studyDepthBtns = document.querySelectorAll('.study-depth-btn');
    if (studyDepthBtns.length > 0) {
        studyDepthBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove selected states from all buttons
                studyDepthBtns.forEach(b => {
                    b.classList.remove('border-2', 'border-primary');
                    b.classList.add('border', 'border-primary/20');
                });
                
                // Add selected state to clicked button
                btn.classList.remove('border', 'border-primary/20');
                btn.classList.add('border-2', 'border-primary');
                
                // Optional: Store the selected depth for later use
                const selectedDepth = btn.getAttribute('data-depth');
                console.log('Selected study depth:', selectedDepth);
            });
        });
    }

});
