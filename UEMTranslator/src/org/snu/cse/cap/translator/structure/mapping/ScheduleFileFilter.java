package org.snu.cse.cap.translator.structure.mapping;

import java.io.File;
import java.io.FileFilter;

import Translators.Constants;

public class ScheduleFileFilter implements FileFilter {
	@Override
	public boolean accept(File pathname) {
		if(pathname.getName().endsWith(Constants.UEMXML_SCHEDULE_PREFIX))
		{
			return true;
		}
		else
		{
			return false;							
		}
	}

}
