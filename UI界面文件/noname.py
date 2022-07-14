# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jan 23 2018)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import wx.adv

###########################################################################
## Class MyFrame1
###########################################################################

class MyFrame1 ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 873,545 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_HIGHLIGHTTEXT ) )
		
		bSizer1 = wx.BoxSizer( wx.VERTICAL )
		
		bSizer2 = wx.BoxSizer( wx.HORIZONTAL )
		
		bSizer3 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_animCtrl1 = wx.adv.AnimationCtrl( self, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition, wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE ) 
		bSizer3.Add( self.m_animCtrl1, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		bSizer2.Add( bSizer3, 9, wx.EXPAND, 5 )
		
		bSizer4 = wx.BoxSizer( wx.VERTICAL )
		
		bSizer9 = wx.BoxSizer( wx.VERTICAL )
		
		sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"参数设置" ), wx.VERTICAL )
		
		sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"视频源" ), wx.VERTICAL )
		
		gSizer1 = wx.GridSizer( 0, 2, 0, 8 )
		
		m_choice1Choices = [ u"摄像头ID_0", u"摄像头ID_1" ]
		self.m_choice1 = wx.Choice( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 95,25 ), m_choice1Choices, 0 )
		self.m_choice1.SetSelection( 0 )
		gSizer1.Add( self.m_choice1, 0, wx.ALL, 5 )
		
		self.camera_button1 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"打开摄像头", wx.DefaultPosition, wx.Size( 95,25 ), 0 )
		gSizer1.Add( self.camera_button1, 0, wx.ALL, 5 )
		
		self.video_button2 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"打开视频文件", wx.DefaultPosition, wx.Size( 95,25 ), 0 )
		gSizer1.Add( self.video_button2, 0, wx.ALL, 5 )
		
		self.off_button3 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"停止", wx.DefaultPosition, wx.Size( 95,25 ), 0 )
		gSizer1.Add( self.off_button3, 0, wx.ALL, 5 )
		
		
		sbSizer2.Add( gSizer1, 1, wx.EXPAND, 5 )
		
		
		sbSizer1.Add( sbSizer2, 1, wx.EXPAND, 5 )
		
		sbSizer3 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"疲劳检测" ), wx.VERTICAL )
		
		bSizer5 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.yawn_checkBox1 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"打哈欠检测", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
		self.yawn_checkBox1.SetValue(True) 
		bSizer5.Add( self.yawn_checkBox1, 0, wx.ALL, 5 )
		
		self.m_staticText3 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"帧数阈值：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )
		bSizer5.Add( self.m_staticText3, 0, wx.ALL, 5 )
		
		m_listBox4Choices = [ u"30", u"32", u"34", u"36", u"38", u"40", u"44", u"46", u"48", u"50" ]
		self.m_listBox4 = wx.ListBox( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 50,24 ), m_listBox4Choices, 0 )
		bSizer5.Add( self.m_listBox4, 0, wx.ALL, 5 )
		
		
		sbSizer3.Add( bSizer5, 1, wx.EXPAND, 5 )
		
		bSizer91 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.blink_checkBox2 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"闭眼检测", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
		self.blink_checkBox2.SetValue(True) 
		bSizer91.Add( self.blink_checkBox2, 0, wx.ALL, 5 )
		
		self.m_staticText31 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"帧数阈值：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText31.Wrap( -1 )
		bSizer91.Add( self.m_staticText31, 0, wx.ALL, 5 )
		
		m_listBox41Choices = [ u"20", u"22", u"24", u"26", u"28", u"30" ]
		self.m_listBox41 = wx.ListBox( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 50,24 ), m_listBox41Choices, 0 )
		bSizer91.Add( self.m_listBox41, 0, wx.ALL, 5 )
		
		
		sbSizer3.Add( bSizer91, 1, wx.EXPAND, 5 )
		
		bSizer11 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.nod_checkBox3 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"点头检测", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
		self.nod_checkBox3.SetValue(True) 
		bSizer11.Add( self.nod_checkBox3, 0, wx.ALL, 5 )
		
		self.m_staticText32 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"帧数阈值：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText32.Wrap( -1 )
		bSizer11.Add( self.m_staticText32, 0, wx.ALL, 5 )
		
		m_listBox42Choices = [ u"10", u"12", u"14", u"16", u"18", u"20" ]
		self.m_listBox42 = wx.ListBox( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 50,24 ), m_listBox42Choices, 0 )
		bSizer11.Add( self.m_listBox42, 0, wx.ALL, 5 )
		
		
		sbSizer3.Add( bSizer11, 1, wx.EXPAND, 5 )
		
		
		sbSizer1.Add( sbSizer3, 3, wx.EXPAND, 5 )
		
		sbSizer4 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"脱岗检测" ), wx.VERTICAL )
		
		bSizer8 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_checkBox4 = wx.CheckBox( sbSizer4.GetStaticBox(), wx.ID_ANY, u"脱岗检测", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_checkBox4.SetValue(True) 
		bSizer8.Add( self.m_checkBox4, 0, wx.ALL, 5 )
		
		self.m_staticText2 = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, u"帧数阈值：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )
		bSizer8.Add( self.m_staticText2, 0, wx.ALL, 5 )
		
		m_listBox2Choices = [ u"5", u"10", u"15", u"20", u"25", u"30" ]
		self.m_listBox2 = wx.ListBox( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 50,24 ), m_listBox2Choices, 0 )
		bSizer8.Add( self.m_listBox2, 0, wx.ALL, 5 )
		
		
		sbSizer4.Add( bSizer8, 1, wx.EXPAND, 5 )
		
		
		sbSizer1.Add( sbSizer4, 1, wx.EXPAND, 5 )
		
		sbSizer10 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"其他设置" ), wx.HORIZONTAL )
		
		gSizer2 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_button4 = wx.Button( sbSizer10.GetStaticBox(), wx.ID_ANY, u"信息重置", wx.DefaultPosition, wx.Size( 95,25 ), 0 )
		gSizer2.Add( self.m_button4, 0, wx.ALL, 5 )
		
		self.m_button5 = wx.Button( sbSizer10.GetStaticBox(), wx.ID_ANY, u"关闭警报", wx.DefaultPosition, wx.Size( 95,25 ), 0 )
		gSizer2.Add( self.m_button5, 0, wx.ALL, 5 )
		
		
		sbSizer10.Add( gSizer2, 1, wx.EXPAND, 5 )
		
		
		sbSizer1.Add( sbSizer10, 1, wx.EXPAND, 5 )
		
		sbSizer7 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"状态检测" ), wx.VERTICAL )
		
		self.m_textCtrl1 = wx.TextCtrl( sbSizer7.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE|wx.TE_READONLY )
		sbSizer7.Add( self.m_textCtrl1, 2, wx.ALL|wx.EXPAND, 5 )
		
		
		sbSizer1.Add( sbSizer7, 2, wx.EXPAND, 5 )
		
		
		bSizer9.Add( sbSizer1, 1, wx.EXPAND, 5 )
		
		
		bSizer4.Add( bSizer9, 1, wx.EXPAND, 5 )
		
		
		bSizer2.Add( bSizer4, 3, wx.EXPAND, 5 )
		
		
		bSizer1.Add( bSizer2, 1, wx.EXPAND, 5 )
		
		
		self.SetSizer( bSizer1 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.camera_button1.Bind( wx.EVT_BUTTON, self.camera_on )
		self.video_button2.Bind( wx.EVT_BUTTON, self.vedio_on )
		self.off_button3.Bind( wx.EVT_BUTTON, self.off )
		self.m_button4.Bind( wx.EVT_BUTTON, self.info_reset )
		self.m_button5.Bind( wx.EVT_BUTTON, self.warning_off )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def camera_on( self, event ):
		event.Skip()
	
	def vedio_on( self, event ):
		event.Skip()
	
	def off( self, event ):
		event.Skip()
	
	def info_reset( self, event ):
		event.Skip()
	
	def warning_off( self, event ):
		event.Skip()
	

