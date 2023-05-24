// Palette de couleurs utilisée par tous les graphiques
var colors = ["#1D507A", "#2F6999", "#66A0D1", "#8FC0E9", "#4682B4"];

// contient les articles de presse, qui doivent être 
// gardés en mémoire même après affichage du graphique
var news_data;


console.log("bonjour");
        
$.ajax({
    url: "/api/news",
    success: display_news
});

console.log("Au revoir");
        
function display_news(result) {
    console.log("Résultat de la requête :", result);
    news_data = result["data"];
    console.log(news_data["articles"].length);
    console.log(news_data["keywords"][0]["word"]);
}

