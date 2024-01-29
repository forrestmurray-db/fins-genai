import ipywidgets as widgets
from IPython.display import display

class ElementHTMLWidget(widgets.HTML):

    def __init__(self, element):
        super().__init__()
        self.element = element
        self.update_html()
    
    def update_html(self):
        html = self.element.metadata.text_as_html if self.element.metadata.text_as_html is not None else '<p>' + str(self.element) + '</p>'
        self.value = html

def display_unstructured_elements(elements):
  # Create widgets for each element in raw_pdf_elements
  element_widgets = [ElementHTMLWidget(element) for element in elements]

  # Display the widgets
  for widget in element_widgets:
      display(widget)