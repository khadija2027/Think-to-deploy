// ================================================
// Main JavaScript - Safran SED
// ================================================

console.log(' Application chargée');

// ============================================
// GESTION DES ONGLETS
// ============================================

function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Retirer la classe active de tous les onglets
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));

            // Ajouter la classe active à l'onglet cliqué
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
}

// ============================================
// INITIALISATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOM chargé, initialisation des composants...');
    initTabs();
    console.log('✅ Tous les composants sont initialisés');
});