package org.snu.cse.cap.translator.structure.mapping;

import java.io.File;
import java.io.FileFilter;

import org.snu.cse.cap.translator.Constants;

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
