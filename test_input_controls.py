"""Regressions for the input layouts and malformed entry observed live."""
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from answer_values import entry_matches
from observer import Observer


class InputControlsTests(unittest.TestCase):
    def test_expression_fields_use_full_root_readback_instead_of_joining_digits(self):
        observer=Observer.__new__(Observer)
        frame=np.full((1000,1600,3),255,np.uint8)
        for x in (500,950):
            cv2.rectangle(frame,(x,450),(x+300,570),(205,205,205),2)
            cv2.putText(frame,'+' if x==500 else '-',(x+120,515),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        with patch('bookwork.BookworkDetector.read_answer_labels',side_effect=[['5+sqrt(10)'],['5-sqrt(10)']]):
            fields=observer.expression_fields(frame)
        self.assertEqual([field['value'] for field in fields],['5+sqrt(10)','5-sqrt(10)'])

    def test_unreadable_selected_math_card_is_never_deselected_as_an_extra(self):
        observer=Observer.__new__(Observer)
        observer.numeric_choice_cards=Mock(return_value=[
            {'value':'','target':[100,300],'selected':True},
            {'value':'81','target':[300,300],'selected':False},
            {'value':'64','target':[500,300],'selected':False}])
        state=observer.numeric_choice_state(None,['81'])
        self.assertFalse(state['complete'])
        self.assertIsNone(state['target'])

    def test_equivalent_malformed_decimal_is_not_accepted_as_readback(self):
        self.assertFalse(entry_matches('00.5','0.5'))
        self.assertFalse(entry_matches('-007.5','-7.5'))
        self.assertTrue(entry_matches('0.5','0.5'))
        self.assertTrue(entry_matches('-7/3',r'-\frac{7}{3}'))

    def test_square_fraction_fields_remain_visible_to_readback(self):
        frame=np.full((1000,1600,3),255,np.uint8)
        for x,width in [(500,100),(850,140)]:
            cv2.rectangle(frame,(x,450),(x+width,555),(205,205,205),2)
        observer=Observer.__new__(Observer)
        observer.read_stacked_fraction=Mock(side_effect=['3/2','-7/3'])
        fields=observer.expression_fields(frame)
        self.assertEqual([field['value'] for field in fields],['3/2','-7/3'])

    def test_selected_numeric_choice_is_bound_to_its_value(self):
        frame=np.full((1000,1600,3),255,np.uint8)
        for index,x in enumerate((350,650,950)):
            cv2.rectangle(frame,(x,500),(x+200,610),(205,205,205),2)
            if index==1:cv2.rectangle(frame,(x,500),(x+200,610),(255,145,77),-1)
        observer=Observer.__new__(Observer)
        observer.keypad_controls=Mock(return_value=None)
        observer.read_stacked_fraction=Mock(side_effect=['1/2','3/2','3/4'])
        state=observer.numeric_choice_state(frame,['3/2'])
        self.assertTrue(state['complete'])
        self.assertEqual(state['cards'][0]['value'],'3/2')

    def test_delete_key_state_distinguishes_empty_from_populated_input(self):
        frame=np.full((300,500,3),240,np.uint8)
        observer=Observer.__new__(Observer);keys={'delete':[250,150]}
        self.assertFalse(observer.keypad_delete_enabled(frame,keys))
        cv2.line(frame,(235,140),(265,160),(20,20,230),2)
        self.assertTrue(observer.keypad_delete_enabled(frame,keys))


if __name__=='__main__':unittest.main()
