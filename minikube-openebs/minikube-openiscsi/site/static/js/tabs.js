/* Tabs JS implementation. Borrowed from Skaffold */
function initTabs() {
  try{
    $('.tab-content').find('.tab-pane').each(function(idx, item) {
      var navTabs = $(this).closest('.code-tabs').find('.nav-tabs'),
          title = escape($(this).attr('title')),
          os = escape($(this).attr('os'));
      navTabs.append('<li class="nav-tab '+os+'"><a href="#" class="nav-tab">'+title+'</a></li>');
    });

    $('.code-tabs ul.nav-tabs').each(function() {
      let tabSelector = getTabSelector(this);
      $(this).find('li'+tabSelector).addClass('active');
    });

    $('.code-tabs .tab-content').each(function() {
      let tabSelector = getTabSelector(this);
      $(this).find('div'+tabSelector).addClass('active');
    })
 
    $('.nav-tabs a').click(function(e){
      e.preventDefault();
      var tab = $(this).parent(),
          tabIndex = tab.index(),
          tabPanel = $(this).closest('.code-tabs'),
          tabPane = tabPanel.find('.tab-pane').eq(tabIndex);
      tabPanel.find('.active').removeClass('active');
      tab.addClass('active');
      tabPane.addClass('active');
    });
  } catch(e) {
    const elements = document.getElementsByClassName("tab-pane");
    for (let element of elements) {
      element.style.display = "block";
      const title = document.createElement("h3");
      title.innerText = element.title;
      title.classList.add("backup-tab-title");
      element.prepend(title);
    }
  }
}

const getTabSelector = currElement => {
    let osSelector = '.'+getUserOS();
    let hasMatchingOSTab = $(currElement).find(osSelector).length;
    return hasMatchingOSTab ? osSelector : ':first';
}

const getUserOS = () => {
    let os = ['Linux', 'Mac', 'Windows'];
    let userAgent = navigator.userAgent;
    for (let currentOS of os) {
      if (userAgent.indexOf(currentOS) !== -1) return currentOS;
    }
    return 'Linux';
}
