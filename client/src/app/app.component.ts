import { Component, inject, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavComponent } from './nav/nav.component';
import { ChatbotService } from './_services/chatbot.service';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
  imports: [RouterOutlet, NavComponent],
})
export class AppComponent implements OnInit {
  private chatbotService = inject(ChatbotService);

  ngOnInit(): void {
    this.chatbotService.create_session().subscribe({});
  }
}
