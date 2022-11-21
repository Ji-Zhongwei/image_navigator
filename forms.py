from wtforms import Form, SelectField, StringField, HiddenField

class DataSearchForm(Form):
    # set the form field for keyword search
    search = StringField('')
    # set hidden field to keep track of images that have been annotated to include in library
    plus_library = HiddenField('')

    # set hidden field to keep track of negative annotations library for training
    minus_library = HiddenField('')

    # set hidden field to keep track of positive annotations for training
    positive = HiddenField('')

    # set hidden field to keep track of negative annotations for training
    negative = HiddenField('')

    # set hidden field to keep track of named facet learners
    facet_names = HiddenField('')

    # sets hidden field to keep track of current facet index
    facet_index = HiddenField('')

    # sets hidden field to keep track of applied facets during search
    selected_facets = HiddenField('')

    # set the hidden field to retain state for view
    view = HiddenField('')

    # set the hidden field to retain state of sorting
    date_ascending = HiddenField('true')

    # set the hidden field to retain state of sorting
    view_sort = HiddenField('')

    # set the session start time
    start_time = HiddenField('')